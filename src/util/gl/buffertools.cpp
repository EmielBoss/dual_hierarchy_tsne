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
#include "dh/util/cu/scan.cuh"

namespace dh::util {

  void BufferTools::init() {
    if (_isInit) {
      return;
    }

    // Initialize shader programs
    {
      _programs(ProgramType::eReduceSumPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceSumPerDatapointIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_per_datapoint_int.comp"));
      _programs(ProgramType::eReduceMinPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceMaxPerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_per_datapoint_float.comp"));
      _programs(ProgramType::eReduceSumFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_float.comp"));
      _programs(ProgramType::eReduceSumIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_sum_int.comp"));
      _programs(ProgramType::eReduceMinFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_float.comp"));
      _programs(ProgramType::eReduceMinVec2Comp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_min_vec2.comp"));
      _programs(ProgramType::eReduceMaxFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_float.comp"));
      _programs(ProgramType::eReduceMaxVec2Comp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_max_vec2.comp"));
      _programs(ProgramType::eReduceCountIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/reduce_count_int.comp"));

      _programs(ProgramType::eRemoveFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_float.comp"));
      _programs(ProgramType::eRemoveIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/remove_int.comp"));

      _programs(ProgramType::eSetIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/set_int.comp"));

      _programs(ProgramType::eFlipIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/flip_int.comp"));

      _programs(ProgramType::eAverageTexturedataComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/average_texturedata.comp"));

      _programs(ProgramType::eOperateFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/operate_float.comp"));

      _programs(ProgramType::eIndexUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/index_uint.comp"));

      _programs(ProgramType::eIndicateIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/indicate_int.comp"));
      _programs(ProgramType::eGetIndicesIntComp).addShader(util::GLShaderType::eCompute, rsrc::get("util/get_indices_int.comp"));

      _programs(ProgramType::eSubsampleUint).addShader(util::GLShaderType::eCompute, rsrc::get("util/subsample_uint.comp"));

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
  T BufferTools::reduce(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, int valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer) {
    glCreateBuffers(_buffersReduce.size() - 1, _buffersReduce.data());
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * sizeof(T), nullptr, 0);
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduced), sizeof(T), nullptr, 0);
    
    if(largeBuffer) {
      glCreateBuffers(1, &_buffersReduce(BufferReduceType::eReducedPerDatapoint));
      glNamedBufferStorage(_buffersReduce(BufferReduceType::eReducedPerDatapoint), n * sizeof(T), nullptr, 0);
      if(std::is_same<T, float>::value) {
        glClearNamedBufferData(_buffersReduce(BufferReduceType::eReducedPerDatapoint), GL_R32F, GL_RED, GL_FLOAT, NULL);
      } else {
        glClearNamedBufferData(_buffersReduce(BufferReduceType::eReducedPerDatapoint), GL_R32I, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
      }

      reducePerDatapoint<T>(bufferToReduce, reductionType, n, _buffersReduce(BufferReduceType::eReducedPerDatapoint), layoutBuffer, neighborsBuffer, selectionBuffer);
      std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eReducedPerDatapoint));
    }

    std::vector<ProgramType> programTypes = {
      std::is_same<T, float>::value ? ProgramType::eReduceSumFloatComp : ProgramType::eReduceSumIntComp,
      std::is_same<T, float>::value ? ProgramType::eReduceMinFloatComp : ProgramType::eReduceMinVec2Comp,
      std::is_same<T, float>::value ? ProgramType::eReduceMaxFloatComp : ProgramType::eReduceMaxVec2Comp,
      ProgramType::eReduceCountIntComp
    };
    dh::util::GLProgram& program = _programs(programTypes[reductionType]);
    program.bind();
  
    program.template uniform<uint>("nPoints", n);
    program.template uniform<bool>("selectedOnly", selectionBuffer > 0);
    if(reductionType == 3) {
      program.template uniform<int>("valueToCount", valueToCount);
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
    if(largeBuffer) { std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eReducedPerDatapoint)); }
    glDeleteBuffers(_buffersReduce.size(), _buffersReduce.data());
    glAssert();
    return reducedValue;
  }

  template<typename T>
  void BufferTools::reducePerDatapoint(GLuint& bufferToReduce, uint reductionType, uint n, GLuint bufferReducedPerDatapoint, GLuint layoutBuffer, GLuint neighborsBuffer, GLuint selectionBuffer) {
    std::vector<ProgramType> programTypes = {
      std::is_same<T, float>::value ? ProgramType::eReduceSumPerDatapointFloatComp : ProgramType::eReduceSumPerDatapointIntComp,
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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bufferReducedPerDatapoint);

    // Dispatch shader
    glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1);
    glFinish();
  }

  template <typename T>
  uint BufferTools::remove(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, GLuint bufferRemoved, bool dynamicStorage) {
    GLuint bufferCumSum;
    glCreateBuffers(1, &bufferCumSum);
    glNamedBufferStorage(bufferCumSum, n * sizeof(uint), nullptr, 0);

    uint nNew;
    util::Scan(selectionBuffer, bufferCumSum, n).comp();
    glGetNamedBufferSubData(bufferCumSum, (n - 1) * sizeof(uint), sizeof(uint), &nNew);

    if(nNew == 0) {
      glDeleteBuffers(1, &bufferCumSum);
      return n;
    }

    bool noDestinationBuffer = bufferRemoved == 0;
    if(noDestinationBuffer) { glCreateBuffers(1, &bufferRemoved); }
    glNamedBufferStorage(bufferRemoved, nNew * d * sizeof(T), nullptr, dynamicStorage ? GL_DYNAMIC_STORAGE_BIT : 0);

    dh::util::GLProgram& program = std::is_same<T, float>::value ? _programs(ProgramType::eRemoveFloatComp) : _programs(ProgramType::eRemoveIntComp);
    program.bind();
  
    program.template uniform<uint>("nPoints", n);
    program.template uniform<uint>("nDims", d);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToRemove);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectionBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferCumSum);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufferRemoved);

    // Dispatch shader
    glDispatchCompute(ceilDiv(n * d, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if(noDestinationBuffer) {
      std::swap(bufferToRemove, bufferRemoved);
      glDeleteBuffers(1, &bufferRemoved);
    }

    glDeleteBuffers(1, &bufferCumSum);
    glAssert();
    return nNew;
  }

  template <typename T>
  void BufferTools::set(GLuint& bufferToSet, uint n, T setVal, T maskVal, GLuint maskBuffer) {
    auto& program = _programs(ProgramType::eSetIntComp);
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
    auto& program = _programs(ProgramType::eFlipIntComp);
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
  
  void BufferTools::operate(uint operationType, GLuint& buffer1, GLuint& buffer2, uint n, GLuint bufferDifference) {
    auto& program = _programs(ProgramType::eOperateFloatComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("n", n);
    program.template uniform<uint>("operationType", operationType);
    program.template uniform<bool>("writeToSeparateBuffer", bufferDifference > 0);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer2);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferDifference);
    glAssert();

    glDispatchCompute(ceilDiv(n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void BufferTools::averageTexturedata(GLuint bufferToAverage, uint n, uint d, uint imgDepth, GLuint maskBuffer, uint maskCount, GLuint bufferAveraged, int maskValue, int maskClass, GLuint subtractorBuffer, bool calcVariance, GLuint labelsBuffer) {
    glCreateBuffers(1, _buffersReduce.data());
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * d * sizeof(float), nullptr, 0);
    
    auto& program = _programs(ProgramType::eAverageTexturedataComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("nPoints", n);
    program.template uniform<uint>("nPointsMasked", maskCount);
    program.template uniform<uint>("nHighDims", d);
    program.template uniform<uint>("imgDepth", imgDepth);
    program.template uniform<bool>("subtract", subtractorBuffer > 0);
    program.template uniform<bool>("calcVariance", calcVariance);
    program.template uniform<int>("maskValue", maskValue);
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

  void BufferTools::index(GLuint& bufferIndices, GLuint& bufferToIndex, uint n) {
    auto& program = _programs(ProgramType::eIndexUintComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("n", n);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToIndex);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferIndices);
    glAssert();

    glDispatchCompute(ceilDiv(n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void BufferTools::getIndices(GLuint& buffer, uint n, uint value, GLuint indicesBuffer) {
    glCreateBuffers(_buffersIndex.size(), _buffersIndex.data());
    std::vector<uint> zeroes(n, 0);
    glNamedBufferStorage(_buffersIndex(BufferIndexType::eIndicated), n * sizeof(uint), zeroes.data(), 0);
    glNamedBufferStorage(_buffersIndex(BufferIndexType::eScanned), n * sizeof(uint), nullptr, 0);

    {
      auto& program = _programs(ProgramType::eIndicateIntComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("n", n);
      program.template uniform<int>("value", value);

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
      util::Scan(_buffersIndex(BufferIndexType::eIndicated), _buffersIndex(BufferIndexType::eScanned), n).comp();
      glGetNamedBufferSubData(_buffersIndex(BufferIndexType::eScanned), (n - 1) * sizeof(uint), sizeof(uint), &count); // Copy the last element of the eScanned buffer (which is the total size) to host
    }

    {
      glNamedBufferStorage(indicesBuffer, count * sizeof(uint), nullptr, 0);

      auto& program = _programs(ProgramType::eGetIndicesIntComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("n", n);
      program.template uniform<int>("value", value);

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
    auto& program = _programs(ProgramType::eSubsampleUint);
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

  // Template instantiations for float, and uint
  template float BufferTools::reduce<float>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, int valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template int BufferTools::reduce<int>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, int valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);
  template glm::vec2 BufferTools::reduce<glm::vec2>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint selectionBuffer, int valueToCount, bool largeBuffer, GLuint layoutBuffer, GLuint neighborsBuffer);

  template void BufferTools::reducePerDatapoint<float>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint bufferReducedPerDatapoint, GLuint layoutBuffer, GLuint neighborsBuffer, GLuint selectionBuffer);
  template void BufferTools::reducePerDatapoint<int>(GLuint& bufferToReduce, uint reductionType, uint n, GLuint bufferReducedPerDatapoint, GLuint layoutBuffer, GLuint neighborsBuffer, GLuint selectionBuffer);

  template uint BufferTools::remove<float>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, GLuint bufferRemoved, bool dynamicStorage);
  template uint BufferTools::remove<int>(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer, GLuint bufferRemoved, bool dynamicStorage);

  template void BufferTools::set<int>(GLuint& bufferToSet, uint n, int setVal, int maskVal, GLuint maskBuffer);

  template void BufferTools::flip<int>(GLuint& bufferToFlip, uint n);
}