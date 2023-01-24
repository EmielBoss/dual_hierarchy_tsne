/*
 * MIT License
 *
 * Copyright (c) 2021 Mark van de Ruit (Delft University of Technology)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <resource_embed/resource_embed.hpp>
#include "dh/util/io.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/cu/inclusive_scan.cuh"

namespace dh::util {

  void BufferTools::init() {
    if (_isInit) {
      return;
    }

    // Initialize shader programs
    {
      _programs(ProgramType::eAccumulatePerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/accumulate_per_datapoint_float.comp"));
      _programs(ProgramType::eAccumulatePerDatapointUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/accumulate_per_datapoint_uint.comp"));
      _programs(ProgramType::eReduceFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_float.comp"));
      _programs(ProgramType::eReduceUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_uint.comp"));

      _programs(ProgramType::eRemoveFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_float.comp"));
      _programs(ProgramType::eRemoveUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_uint.comp"));

      _programs(ProgramType::eSetUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/set_uint.comp"));

      _programs(ProgramType::eFlipUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/flip_uint.comp"));

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }
    
    _isInit = true;
  }

  void BufferTools::dstr() {
    if (_isInit) {
      return;
    }

    _isInit = false;
  }

  BufferTools::BufferTools() : _isInit(false) { }

  BufferTools::~BufferTools() {
    if (_isInit) {
      dstr();
    }
  }
  
  template<typename T>
  T BufferTools::reduce(GLuint& bufferToReduce, uint n, T countVal, bool largeBuffer, GLuint selectionBuffer, GLuint layoutBuffer, GLuint neighborsBuffer) {
    glCreateBuffers(_buffersReduce.size(), _buffersReduce.data());
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * sizeof(T), nullptr, 0);
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduced), sizeof(T), nullptr, 0);

    if(largeBuffer) {
      glNamedBufferStorage(_buffersReduce(BufferReduceType::eAccumulationPerDatapoint), n * sizeof(T), nullptr, 0);

      dh::util::GLProgram& program = std::is_same<T, float>::value
                                    ? _programs(ProgramType::eAccumulatePerDatapointFloatComp)
                                    : _programs(ProgramType::eAccumulatePerDatapointUintComp);
      program.bind();

      program.template uniform<uint>("nPoints", n);
      program.template uniform<bool>("selectedOnly", selectionBuffer > 0);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layoutBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, neighborsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufferToReduce);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1);
      glAssert();
      std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint));
    }

    dh::util::GLProgram& program = std::is_same<T, float>::value ? _programs(ProgramType::eReduceFloatComp) : _programs(ProgramType::eReduceUintComp);
    program.bind();
  
    program.template uniform<uint>("nPoints", n);
    program.template uniform<bool>("sumAll", countVal == -1);
    program.template uniform<T>("countVal", countVal);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToReduce);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersReduce(BufferReduceType::eReduce));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersReduce(BufferReduceType::eReduced));

    // Dispatch shader
    program.template uniform<uint>("iter", 0);
    glDispatchCompute(128, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.template uniform<uint>("iter", 1);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    T reducedValue = 0;
    glGetNamedBufferSubData(_buffersReduce(BufferReduceType::eReduced), 0, sizeof(T), &reducedValue);
    if(largeBuffer) { std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint)); }
    glDeleteBuffers(_buffersReduce.size(), _buffersReduce.data());
    glAssert();
    return reducedValue;
  }

  template <typename T>
  uint BufferTools::remove(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer) {
    glCreateBuffers(_buffersRemove.size(), _buffersRemove.data());
    glNamedBufferStorage(_buffersRemove(BufferRemoveType::eCumSum), n * sizeof(T), nullptr, 0);

    uint nNew;
    {
      util::InclusiveScan scan(selectionBuffer, _buffersRemove(BufferRemoveType::eCumSum), n);
      scan.comp();
      glGetNamedBufferSubData(_buffersRemove(BufferRemoveType::eCumSum), (n - 1) * sizeof(uint), sizeof(uint), &nNew);
    }

    if(nNew > 0) {
      glNamedBufferStorage(_buffersRemove(BufferRemoveType::eRemoved), nNew * d * sizeof(T), nullptr, 0);

      dh::util::GLProgram& program = std::is_same<T, float>::value ? _programs(ProgramType::eRemoveFloatComp) : _programs(ProgramType::eRemoveUintComp);
      program.bind();
    
      program.template uniform<uint>("nPoints", n);
      program.template uniform<uint>("nDims", d);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToRemove);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersRemove(BufferRemoveType::eCumSum));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersRemove(BufferRemoveType::eRemoved));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n * d, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      std::swap(bufferToRemove, _buffersRemove(BufferRemoveType::eRemoved));

      glDeleteBuffers(_buffersRemove.size(), _buffersRemove.data());
      glAssert();
      return nNew;
    }
    else {
      glDeleteBuffers(_buffersRemove.size(), _buffersRemove.data());
      return n;
    }
  }

  template <typename T>
  void BufferTools::set(GLuint& bufferToSet, uint n, T setVal, T maskVal, GLuint maskBuffer) {
    auto& program = _programs(ProgramType::eSetUintComp);
    program.bind();

    // Set uniform
    program.template uniform<uint>("nPoints", n);
    program.template uniform<T>("maskVal", maskVal);
    program.template uniform<T>("setVal", setVal);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, maskBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferToSet);

    // Dispatch shader
    glDispatchCompute(ceilDiv(n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  template <typename T>
  void BufferTools::flip(GLuint& bufferToFlip, uint n) {
    auto& program = _programs(ProgramType::eFlipUintComp);
    program.bind();

    // Set uniform
    program.template uniform<uint>("nPoints", n);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToFlip);

    // Dispatch shader
    glDispatchCompute(ceilDiv(n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  // Template instantiations for float, and uint
  template float BufferTools::reduce<float>(GLuint& bufferToReduce, uint n, float countVal, bool largeBuffer, GLuint selectionBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template uint BufferTools::reduce<uint>(GLuint& bufferToReduce, uint n, uint countVal, bool largeBuffer, GLuint selectionBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template uint BufferTools::remove<float>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer);
  template uint BufferTools::remove<uint>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer);
  template void BufferTools::set<uint>(GLuint& bufferToSet, uint n, uint setVal, uint maskVal, GLuint maskBuffer);
  template void BufferTools::flip<uint>(GLuint& bufferToFlip, uint n);
}