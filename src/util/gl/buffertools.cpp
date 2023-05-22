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
      _programs(ProgramType::eReduceSumPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceSumPerDatapointUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_per_datapoint_uint.comp"));
      _programs(ProgramType::eReduceMinPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceMaxPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceSumFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_float.comp"));
      _programs(ProgramType::eReduceSumUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_uint.comp"));
      _programs(ProgramType::eReduceMinFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_float.comp"));
      _programs(ProgramType::eReduceMinVec2Comp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_vec2.comp"));
      _programs(ProgramType::eReduceMaxFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_float.comp"));
      _programs(ProgramType::eReduceMaxVec2Comp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_vec2.comp"));
      _programs(ProgramType::eReduceCountUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_count_uint.comp"));

      _programs(ProgramType::eRemoveFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_float.comp"));
      _programs(ProgramType::eRemoveUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_uint.comp"));

      _programs(ProgramType::eSetUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/set_uint.comp"));

      _programs(ProgramType::eFlipUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/flip_uint.comp"));

      _programs(ProgramType::eAverageTexturedataComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/average_texturedata.comp"));

      _programs(ProgramType::eDifferenceComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/difference.comp"));

      _programs(ProgramType::eIndicateComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/indicate.comp"));
      _programs(ProgramType::eIndexComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/index.comp"));

      _programs(ProgramType::eSubsample).addShader(util::GLShaderType::eCompute, rsrc::get("util/subsample.comp"));

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
  T BufferTools::reduce(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, uint valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer) {
    glCreateBuffers(_buffersReduce.size(), _buffersReduce.data());
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * sizeof(T), nullptr, 0);
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduced), sizeof(T), nullptr, 0);
    
    if(largeBuffer) {
      glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduceSumPerDatapoint), n * sizeof(T), nullptr, 0);
      if(std::is_same<T, float>::value) {
        glClearNamedBufferData(_buffersReduce(BufferReduceType::eReduceSumPerDatapoint), GL_R32F, GL_RED, GL_FLOAT, nullptr);
      } else {
        glClearNamedBufferData(_buffersReduce(BufferReduceType::eReduceSumPerDatapoint), GL_R32I, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      }

      std::vector<ProgramType> programTypes = {
        std::is_same<T, float>::value ? ProgramType::eReduceSumPerDatapointFloatComp : ProgramType::eReduceSumPerDatapointUintComp,
        ProgramType::eReduceMinPerDatapointFloatComp,
        ProgramType::eReduceMaxPerDatapointFloatComp
      };
      dh::util::GLProgram& program = _programs(programTypes[reductionType]);
      program.bind();

      program.template uniform<uint>("nPoints", n);
      program.template uniform<bool>("selectedOnly", selectionBuffer > 0);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layoutBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, neighborsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufferToReduce);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersReduce(BufferReduceType::eReduceSumPerDatapoint));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1);
      glFinish();
      std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eReduceSumPerDatapoint));
    }

    std::vector<ProgramType> programTypes = {
      std::is_same<T, float>::value ? ProgramType::eReduceSumFloatComp : ProgramType::eReduceSumUintComp,
      std::is_same<T, float>::value ? ProgramType::eReduceMinFloatComp : ProgramType::eReduceMinVec2Comp,
      std::is_same<T, float>::value ? ProgramType::eReduceMaxFloatComp : ProgramType::eReduceMaxVec2Comp,
      ProgramType::eReduceCountUintComp
    };
    dh::util::GLProgram& program = _programs(programTypes[reductionType]);
    program.bind();
  
    program.template uniform<uint>("nPoints", n);
    program.template uniform<bool>("selectedOnly", selectionBuffer > 0);
    if(reductionType == 3) {
      program.template uniform<uint>("valueToCount", valueToCount);
    }

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToReduce);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersReduce(BufferReduceType::eReduce));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersReduce(BufferReduceType::eReduced));

    // Dispatch shader
    program.template uniform<uint>("iter", 0);
    glDispatchCompute(128, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.template uniform<uint>("iter", 1);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    T reducedValue;
    glGetNamedBufferSubData(_buffersReduce(BufferReduceType::eReduced), 0, sizeof(T), &reducedValue);
    if(largeBuffer) { std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eReduceSumPerDatapoint)); }
    glDeleteBuffers(_buffersReduce.size(), _buffersReduce.data());
    glAssert();
    return reducedValue;
  }

  template <typename T>
  uint BufferTools::remove(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, bool dynamicStorage) {
    glCreateBuffers(_buffersRemove.size(), _buffersRemove.data());
    glNamedBufferStorage(_buffersRemove(BufferRemoveType::eCumSum), n * sizeof(T), nullptr, 0);

    uint nNew;
    {
      util::InclusiveScan scan(selectionBuffer, _buffersRemove(BufferRemoveType::eCumSum), n);
      scan.comp();
      glGetNamedBufferSubData(_buffersRemove(BufferRemoveType::eCumSum), (n - 1) * sizeof(uint), sizeof(uint), &nNew);
    }

    if(nNew > 0) {
      glNamedBufferStorage(_buffersRemove(BufferRemoveType::eRemoved), nNew * d * sizeof(T), nullptr, dynamicStorage ? GL_DYNAMIC_STORAGE_BIT : 0);

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

  void BufferTools::averageTexturedata(GLuint bufferToAverage, uint n, uint d, uint imgDepth, GLuint maskBuffer, uint maskValue, uint maskCount, GLuint bufferAveraged, GLuint subtractorBuffer, bool calcVariance, int maskClass, GLuint labelsBuffer) {
    glCreateBuffers(1, _buffersReduce.data());
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * d * sizeof(float), nullptr, 0);
    
    auto& program = _programs(ProgramType::eAverageTexturedataComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("nPoints", n);
    program.template uniform<uint>("nPointsMasked", maskCount);
    program.template uniform<uint>("nHighDims", d);
    program.template uniform<uint>("imgDepth", imgDepth);
    program.template uniform<uint>("maskValue", maskValue);
    program.template uniform<bool>("subtract", subtractorBuffer > 0);
    program.template uniform<bool>("calcVariance", calcVariance);
    program.template uniform<int>("maskClass", maskClass);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToAverage);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, maskBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, subtractorBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersReduce(BufferReduceType::eReduce));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bufferAveraged);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, labelsBuffer);
    glAssert();

    program.template uniform<uint>("iter", 0);
    glDispatchCompute(128, d, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.template uniform<uint>("iter", 1);
    glDispatchCompute(1, d, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, _buffersReduce.data());
    glAssert();
  }

  void BufferTools::difference(GLuint& buffer1, GLuint& buffer2, uint n, GLuint& bufferDifference) {
    auto& program = _programs(ProgramType::eDifferenceComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("n", n);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferDifference);
    glAssert();

    glDispatchCompute(ceilDiv(n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void BufferTools::index(GLuint& buffer, uint n, uint value, GLuint indicesBuffer) {
    glCreateBuffers(_buffersIndex.size(), _buffersIndex.data());
    std::vector<uint> zeroes(n, 0);
    glNamedBufferStorage(_buffersIndex(BufferIndexType::eIndicated), n * sizeof(uint), zeroes.data(), 0);
    glNamedBufferStorage(_buffersIndex(BufferIndexType::eScanned), n * sizeof(uint), nullptr, 0);

    {
      auto& program = _programs(ProgramType::eIndicateComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("n", n);
      program.template uniform<uint>("value", value);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersIndex(BufferIndexType::eIndicated));
      glAssert();

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    uint count;
    {
      util::InclusiveScan(_buffersIndex(BufferIndexType::eIndicated), _buffersIndex(BufferIndexType::eScanned), n).comp();
      glGetNamedBufferSubData(_buffersIndex(BufferIndexType::eScanned), (n - 1) * sizeof(uint), sizeof(uint), &count); // Copy the last element of the eScanned buffer (which is the total size) to host
    }

    {
      glNamedBufferStorage(indicesBuffer, count * sizeof(uint), nullptr, 0);

      auto& program = _programs(ProgramType::eIndexComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("n", n);
      program.template uniform<uint>("value", value);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersIndex(BufferIndexType::eScanned));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, indicesBuffer);
      glAssert();

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }
  }

  void BufferTools::subsample(GLuint& bufferToSubsample, uint n, uint every, uint outOf, GLuint bufferSubsampled) {
    glNamedBufferStorage(bufferSubsampled, n * sizeof(uint), nullptr, 0);

    {
      auto& program = _programs(ProgramType::eSubsample);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("n", n);
      program.template uniform<uint>("everyIndex", every - 1);
      program.template uniform<uint>("outOf", outOf);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToSubsample);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferSubsampled);
      glAssert();

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }
  }

  // Template instantiations for float, and uint
  template float BufferTools::reduce<float>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, uint valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template uint BufferTools::reduce<uint>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, uint valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template glm::vec2 BufferTools::reduce<glm::vec2>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, uint valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template uint BufferTools::remove<float>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, bool dynamicStorage);
  template uint BufferTools::remove<uint>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, bool dynamicStorage);
  template void BufferTools::set<uint>(GLuint& bufferToSet, uint n, uint setVal, uint maskVal, GLuint maskBuffer);
  template void BufferTools::flip<uint>(GLuint& bufferToFlip, uint n);
}