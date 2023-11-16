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
#include "dh/sne/components/similarities.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/io.hpp" //
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/cu/scan.cuh"
#include "dh/util/cu/knn.cuh"
#include <algorithm>
#include <cmath>
#include <typeinfo> //
#include <numeric> //
#include <imgui.h> //
#include <imgui_impl_glfw.h> //
#include <imgui_impl_opengl3.h> //
#include <implot.h> //
#include <set> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");

  float Similarities::average(std::vector<float> vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.f) / vec.size();
  }
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const float* dataPtr, Params* params)
  : _isInit(false), _dataPtr(dataPtr), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/similarities.comp"));
      _programs(ProgramType::eExpandComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/expand.comp"));
      _programs(ProgramType::eLayoutComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/layout.comp"));
      _programs(ProgramType::eNeighborsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors.comp"));
      _programs(ProgramType::eNeighborsSortComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors_sort.comp"));
      _programs(ProgramType::eL1DistancesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/L1_distances.comp"));
      _programs(ProgramType::eWeighSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerDatapointComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_datapoint.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeRatioComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_ratio.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeRangeComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_range.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeResembleComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_resemble.comp"));
      _programs(ProgramType::eSubDistancesL1Comp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/subdistances.comp"));
      _programs(ProgramType::eUpdateSizesInter).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/update_sizes_inter.comp"));
      _programs(ProgramType::eUpdateSizesIntra).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/update_sizes_intra.comp"));
      _programs(ProgramType::eCopyOldStuff).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/copy_old_stuff.comp"));
      _programs(ProgramType::eFillNewNeighborsInter).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/fill_new_neighbors_inter.comp"));
      _programs(ProgramType::eFillNewNeighborsIntra).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/fill_new_neighbors_intra.comp"));
      _programs(ProgramType::eFillNewDistances).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/fill_new_distances.comp"));
      _programs(ProgramType::eFillNewSimilarities).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/fill_new_similarities.comp"));
      _programs(ProgramType::eSymmetrize).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/symmetrize.comp"));

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Create and initialize buffers
    glCreateBuffers(_buffers.size(), _buffers.data());
    {
      const std::vector<float> ones(_params->nHighDims, 1.0f);
      std::vector<float> data;
      data.assign(dataPtr, dataPtr + _params->n * _params->nHighDims);

      glNamedBufferStorage(_buffers(BufferType::eDataset), _params->n * _params->nHighDims * sizeof(float), data.data(), 0); // Original dataset
      glNamedBufferStorage(_buffers(BufferType::eLayout), _params->n * 2 * sizeof(uint), nullptr, 0); // n structs of two uints; the first is its expanded neighbor set offset (eScan[i - 1]), the second is its expanded neighbor set size (eScan[i] - eScan[i - 1])
      glNamedBufferStorage(_buffers(BufferType::eAttributeWeights), _params->nHighDims * sizeof(float), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glAssert();
    }

    _isInit = true;
    Logger::rest() << prefix << "Initialized";

    dh::util::BufferTools::instance().init();
  }

  Similarities::~Similarities() {
    if (isInit()) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  Similarities::Similarities(Similarities&& other) noexcept {
    swap(*this, other);
  }

  Similarities& Similarities::operator=(Similarities&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void Similarities::comp() {
    runtimeAssert(isInit(), "Similarities::comp() called without proper initialization");

    // Create and initialize temporary buffer objects
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    {
      std::vector<uint> zeroes(_params->n * _params->k, 0);
      glNamedBufferStorage(_buffersTemp(BufferTempType::eDistancesL2), _params->n * _params->k * sizeof(float), nullptr, 0); // n * k floats of neighbor distances; every k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eNeighbors), _params->n * _params->k * sizeof(uint), nullptr, 0); // n * k uints of neighbor indices (ranging from 0 to n-1); every k'th element is vector index itself (so it's actually k-1 NN)
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSimilarities), _params->n * _params->k * sizeof(float), zeroes.data(), 0); // n * k floats of neighbor similarities; every k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSizes), _params->n * sizeof(uint), zeroes.data(), 0); // n uints of (expanded) neighbor set sizes; every element is k-1 plus its number of "unregistered neighbors" that have it as neighbor but that it doesn't reciprocate
      glNamedBufferStorage(_buffersTemp(BufferTempType::eScan), _params->n * sizeof(uint), nullptr, 0); // Prefix sum/inclusive scan over expanded neighbor set sizes (eSizes). (This should be a temp buffer, but that yields an error)
      glNamedBufferStorage(_buffersTemp(BufferTempType::eCounts), _params->n * sizeof(uint), zeroes.data(), 0);
    }
    
    // Progress bar for logging steps of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");
    progressBar.setPostfix("Performing KNN search");
    progressBar.setProgress(0.0f);

    // 1.
    // Compute approximate KNN of each point, delegated to FAISS
    // Produces a fixed number of neighbors
    {
      util::KNN knn(
        _buffers(BufferType::eDataset),
        _buffersTemp(BufferTempType::eDistancesL2),
        _buffersTemp(BufferTempType::eNeighbors),
        _params->n, _params->k, _params->nHighDims);
      knn.comp();
    }

    // Update progress bar
    progressBar.setPostfix("Performing similarity computation");
    progressBar.setProgress(1.0f / 6.0f);
    
    // 2.
    // Compute similarities over generated KNN. This is pretty much a direct copy of the formulation used in BH-SNE, and seems to also be used in CUDA-tSNE.
    {
      auto& timer = _timers(TimerType::eSimilaritiesComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eSimilaritiesComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("kNeighbors", _params->k);
      program.template uniform<float>("perplexity", _params->perplexity);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eDistancesL2));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Symmetrizing KNN data");
    progressBar.setProgress(2.0f / 6.0f);

    // 3.
    // Expand KNN data so it becomes symmetric. That is, every neigbor referred by a point itself refers to that point as a neighbor.
    // Actually just fills eSizes, which, for each element, is _params->k-1+the number of "unregistered neighbors"; datapoints that have it as a neighbor but which it doesn't reciprocate
    {
      auto& timer = _timers(TimerType::eExpandComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eExpandComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("kNeighbors", _params->k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Allocating buffers");
    progressBar.setProgress(3.0f / 6.0f);

    // 4.
    // Determine sizes of expanded neighborhoods in memory through prefix sum (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath
    util::Scan(_buffersTemp(BufferTempType::eSizes), _buffersTemp(BufferTempType::eScan), _params->n).comp();
    glGetNamedBufferSubData(_buffersTemp(BufferTempType::eScan), (_params->n - 1) * sizeof(uint), sizeof(uint), &_symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host

    // Update progress bar
    progressBar.setPostfix("Computing layout");
    progressBar.setProgress(4.0f / 6.0f);

    // 5.
    // Fill layout buffer
    {
      auto& timer = _timers(TimerType::eLayoutComp);
      timer.tick();

      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Symmetrizing similarities");
    progressBar.setProgress(5.0f / 6.0f);

    // Initialize permanent buffer objects
    std::vector<float> zeroes(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffers(BufferType::eNeighbors), _symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), _symmetricSize * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT); // Corresponding similarities
    glNamedBufferStorage(_buffers(BufferType::eDistancesL1), _symmetricSize * sizeof(float), zeroes.data(), 0); // Corresponding distances

    // 6.
    // Generate expanded similarities and neighbor buffers, symmetrized and ready for use during the minimization
    {
      auto& timer = _timers(TimerType::eNeighborsComp);
      timer.tick();

      auto &program = _programs(ProgramType::eNeighborsComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("kNeighbors", _params->k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n * (_params->k-1), 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 7.
    // Sort neighbours within each KNN set (and corresponding similarities)
    {
      auto &program = _programs(ProgramType::eNeighborsSortComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }
    
    // 8.
    // Calculate L1 distances
    {
      auto &program = _programs(ProgramType::eL1DistancesComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDistancesL1));

      // Dispatch shader in batches of batchSize attriibutes
      uint batchSize = 5;
      for(uint b = 0; b * batchSize < _params->nHighDims; ++b) {
        program.template uniform<uint>("batchBegin", b * batchSize);
        program.template uniform<uint>("batchEnd", std::min((b+1) * batchSize, (uint) _params->nHighDims));
        glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
        glFinish();
        glAssert();
      }
    }
    
    // Keep backup of similarities in eSimilaritiesOriginal, because eSimilarities may get changed
    glNamedBufferStorage(_buffers(BufferType::eSimilaritiesOriginal), _symmetricSize * sizeof(float), nullptr, 0);
    glCopyNamedBufferSubData(_buffers(BufferType::eSimilarities), _buffers(BufferType::eSimilaritiesOriginal), 0, 0, _symmetricSize * sizeof(float));

    // Update progress bar
    progressBar.setPostfix("Done!");
    progressBar.setProgress(1.0f);

    // Delete temporary buffers
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    glAssert();

    // Output memory use of persistent OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::curt() << prefix << "Completed, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Poll twice so front/back timers are swapped
    glPollTimers(_timers.size(), _timers.data());
    glPollTimers(_timers.size(), _timers.data());
  }

  void Similarities::recomp(GLuint selectionBufferHandle, float perplexity, uint k) {
    _params->n = dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eDataset), _params->n, _params->nHighDims, selectionBufferHandle);
    _params->perplexity = perplexity;
    _params->k = k;
    glDeleteBuffers(1, &_buffers(BufferType::eNeighbors));
    glCreateBuffers(1, &_buffers(BufferType::eNeighbors));
    glDeleteBuffers(1, &_buffers(BufferType::eSimilarities));
    glCreateBuffers(1, &_buffers(BufferType::eSimilarities));
    glDeleteBuffers(1, &_buffers(BufferType::eSimilaritiesOriginal));
    glCreateBuffers(1, &_buffers(BufferType::eSimilaritiesOriginal));
    glDeleteBuffers(1, &_buffers(BufferType::eDistancesL1));
    glCreateBuffers(1, &_buffers(BufferType::eDistancesL1));
    comp();
  }

  // Renormalizing the similarities
  void Similarities::renormalizeSimilarities(GLuint selectionBufferHandle) {
    float simSumOrg = dh::util::BufferTools::instance().reduce<float>(_buffers(BufferType::eSimilaritiesOriginal), 0, _params->n, selectionBufferHandle, -1, true, _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    float simSumNew = dh::util::BufferTools::instance().reduce<float>(_buffers(BufferType::eSimilarities), 0, _params->n, selectionBufferHandle, -1, true, _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    float factor = simSumOrg / simSumNew;
    weighSimilarities(factor, selectionBufferHandle);
  }

  void Similarities::weighSimilarities(float weight, GLuint selectionBufferHandle, bool interOnly) {
    auto &program = _programs(ProgramType::eWeighSimilaritiesComp);
    program.bind();

    program.template uniform<uint>("nPoints", _params->n);
    program.template uniform<float>("weight", weight);
    program.template uniform<bool>("selectedOnly", selectionBufferHandle > 0);
    program.template uniform<bool>("interOnly", interOnly);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilarities));

    // Dispatch shader
    glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void Similarities::weighSimilaritiesPerDatapoint(GLuint weightsBuffer, GLuint selectionBufferHandle, bool interOnly) {
    auto &program = _programs(ProgramType::eWeighSimilaritiesPerDatapointComp);
    program.bind();

    program.template uniform<uint>("nPoints", _params->n);
    program.template uniform<bool>("selectedOnly", selectionBufferHandle > 0);
    program.template uniform<bool>("interOnly", interOnly);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, weightsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eLayout));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighbors));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eSimilarities));

    // Dispatch shader
    glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void Similarities::weighSimilaritiesPerAttributeRatio(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected) {
    if(weightedAttributeIndices.size() == 0) { return; }
    
    // Create and initialize temp buffers
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> setvec(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), weightedAttributeIndices.size() * sizeof(uint), setvec.data(), 0);

    // Weighting the similarities
    {
      auto &program = _programs(ProgramType::eWeighSimilaritiesPerAttributeRatioComp);
      program.bind();

      float mult = (_params->nHighDims / weightedAttributeIndices.size()) / (5.f * (1.f - weightedAttributeIndices.size() / _params->nHighDims) + 1);
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      program.template uniform<uint>("nWeightedAttribs", weightedAttributeIndices.size());
      program.template uniform<float>("multiplier", mult);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eWeightedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttributeWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glAssert();
    }

    renormalizeSimilarities(selectionBufferHandle);
    
    glAssert();
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
  }

  void Similarities::weighSimilaritiesPerAttributeRange(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected) {
    if(weightedAttributeIndices.size() == 0) { return; }

    // Create and initialize temp buffers
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> setvec(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), weightedAttributeIndices.size() * sizeof(uint), setvec.data(), 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eSubDistancesL1), _symmetricSize * sizeof(float), nullptr, 0);
    glClearNamedBufferData(_buffersTemp(BufferTempType::eSubDistancesL1), GL_R32F, GL_RED, GL_FLOAT, NULL); // Initialize with all zeros

    // Obtaining the subdistances across the weighted attributes
    {
      auto &program = _programs(ProgramType::eSubDistancesL1Comp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      program.template uniform<uint>("nWeightedAttribs", weightedAttributeIndices.size());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eWeightedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttributeWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffersTemp(BufferTempType::eSubDistancesL1));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glAssert();
    }

    float distMin = dh::util::BufferTools::instance().reduce<float>(_buffersTemp(BufferTempType::eSubDistancesL1), 1, _params->n, selectionBufferHandle, -1, true, _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    float distMax = dh::util::BufferTools::instance().reduce<float>(_buffersTemp(BufferTempType::eSubDistancesL1), 2, _params->n, selectionBufferHandle, -1, true, _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    
    // Weighting the similarities
    {
      auto &program = _programs(ProgramType::eWeighSimilaritiesPerAttributeRangeComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<float>("distMin", distMin);
      program.template uniform<float>("distRangeInv", 1.f / (distMax - distMin));

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eWeightedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersTemp(BufferTempType::eSubDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glAssert();
    }

    renormalizeSimilarities(selectionBufferHandle);

    glAssert();
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
  }

  void Similarities::weighSimilaritiesPerAttributeResemble(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, std::vector<uint> archetypeIndices, std::vector<uint> archetypeLabels) {
    if(archetypeLabels.size() < 1) { return; }
    
    // Create and initialize temp buffers
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> attributeIndices;
    if(weightedAttributeIndices.size() > 0) {
      attributeIndices = std::vector<uint>(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    } else {
      attributeIndices = std::vector<uint>(_params->nHighDims);
      std::iota(attributeIndices.begin(), attributeIndices.end(), 0);
      glClearNamedBufferData(_buffers(BufferType::eAttributeWeights), GL_R32F, GL_RED, GL_FLOAT, NULL);
    }
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), attributeIndices.size() * sizeof(uint), attributeIndices.data(), 0);

    uint nArchetypes = archetypeLabels.size();
    glNamedBufferStorage(_buffersTemp(BufferTempType::eArchetypeLabels), nArchetypes * sizeof(uint), archetypeLabels.data(), 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eArchetypesIndices), nArchetypes * sizeof(uint), archetypeIndices.data(), 0);

    // Weighting the similarities
    {
      auto &program = _programs(ProgramType::eWeighSimilaritiesPerAttributeResembleComp);
      program.bind();
      glAssert();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      program.template uniform<uint>("nWeightedAttribs", attributeIndices.size());
      program.template uniform<uint>("nArchetypes", nArchetypes);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eArchetypesIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eArchetypeLabels));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersTemp(BufferTempType::eWeightedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eAttributeWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glAssert();
    }

    renormalizeSimilarities(selectionBufferHandle);
    
    if(weightedAttributeIndices.size() == 0) {
      const std::vector<float> ones(_params->nHighDims, 1.f);
      glClearNamedBufferData(_buffers(BufferType::eAttributeWeights), GL_R32F, GL_RED, GL_FLOAT, ones.data());
    }
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    glAssert();
  }

  void Similarities::editLinksInter(GLuint selectionBufferHandle, std::vector<uint> selectionCounts, bool addOrRemove, float siphonRate) {
    // Calculates upper bound based on the order of magnitude of the selection size. Examples: upperBound(7) = 0.9,  upperBound(43) = 0.9, upperBound(123) = 0.99, upperBound(12368) = 0.9999
    auto omissionRatioUpperBound = [](uint n) {
      float nDigits = std::floor(std::log2((float) n) / std::log2(10)) + 1;
      float multiplier = std::pow(10.f, nDigits - 1);
      float bound = floor(0.999999999 * multiplier) / multiplier;
      return std::max(bound, 0.9f);
    };

    if(selectionCounts[0] == 0 || selectionCounts[1] == 0) { return; }

    // Get indices of selected datapoints into corresponding buffers
    glCreateBuffers(_buffersFuse.size(), _buffersFuse.data());
    dh::util::BufferTools::instance().getIndices(selectionBufferHandle, _params->n, 1, _buffersFuse(BufferFuseType::eIndicesSelectionPrimary));
    dh::util::BufferTools::instance().getIndices(selectionBufferHandle, _params->n, 2, _buffersFuse(BufferFuseType::eIndicesSelectionSecondary));
    uint selectionCount = selectionCounts[0] + selectionCounts[1];
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eIndicesSelection), selectionCount * sizeof(uint), nullptr, 0);
    glCopyNamedBufferSubData(_buffersFuse(BufferFuseType::eIndicesSelectionPrimary), _buffersFuse(BufferFuseType::eIndicesSelection), 0, 0, selectionCounts[0] * sizeof(uint));
    glCopyNamedBufferSubData(_buffersFuse(BufferFuseType::eIndicesSelectionSecondary), _buffersFuse(BufferFuseType::eIndicesSelection), 0, selectionCounts[0] * sizeof(uint), selectionCounts[1] * sizeof(uint));
    
    // eSizes contains every second element of eLayout
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSizes), _params->n * sizeof(uint), nullptr, 0);
    dh::util::BufferTools::instance().subsample(_buffers(BufferType::eLayout), _params->n, 2, 2, _buffersFuse(BufferFuseType::eSizes));

    // Get preexisting similarity both total and per datapoint
    std::vector<float> zeroes(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _params->n * sizeof(float), zeroes.data(), 0);
    dh::util::BufferTools::instance().reducePerDatapoint<float>(_buffers(BufferType::eSimilarities), 0, _params->n, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));

    // Get new nearest neighbor set sizes
    {
      auto& program = _programs(ProgramType::eUpdateSizesInter);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eSizes));
      glAssert();

      // Set uniforms
      program.template uniform<bool>("addOrRemove", addOrRemove);

      for(uint i = 0; i < 2; ++i) {
        uint nSelected = selectionCounts[i];
        uint nSelectedOther = selectionCounts[1 - i];
        program.template uniform<uint>("nSelected", nSelected);
        program.template uniform<uint>("nSelectedOther", nSelectedOther);
        program.template uniform<uint>("offsetOther", (1 - i) * selectionCounts[0]);
        program.template uniform<float>("omissionRatio", std::clamp(1.f - (float) _params->k / nSelectedOther, 0.f, omissionRatioUpperBound(nSelectedOther)));
        program.template uniform<float>("omissionRatioOther", std::clamp(1.f - (float) _params->k / nSelected, 0.f, omissionRatioUpperBound(nSelected)));
        glDispatchCompute(ceilDiv(nSelected, 256u / 32u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      glAssert();
    }

    glNamedBufferStorage(_buffersFuse(BufferFuseType::eScan), _params->n * sizeof(uint), nullptr, 0);
    util::Scan(_buffersFuse(BufferFuseType::eSizes), _buffersFuse(BufferFuseType::eScan), _params->n).comp(); // Determine sizes of expanded neighborhoods in memory through prefix sum (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath
    glGetNamedBufferSubData(_buffersFuse(BufferFuseType::eScan), (_params->n - 1) * sizeof(uint), sizeof(uint), &_symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eLayout), _params->n * 2 * sizeof(uint), nullptr, 0);

    // Fill layout buffer
    {
      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersFuse(BufferFuseType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    // Create buffers for new links, will be swapped later
    std::vector<uint> zeroesU(_symmetricSize, 0);
    std::vector<float> zeroesF(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eNeighbors), _symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimilarities), _symmetricSize * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT); // Corresponding similarities
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimilaritiesOriginal), _symmetricSize * sizeof(float), nullptr, 0); // Corresponding similarities
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eDistancesL1), _symmetricSize * sizeof(float), zeroesF.data(), 0); // Corresponding distances
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eDistancesL2), _symmetricSize * sizeof(float), zeroesF.data(), 0); // Corresponding distances
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eCounts), _params->n * sizeof(uint), zeroesU.data(), 0);

    {
      auto& program = _programs(ProgramType::eCopyOldStuff);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<bool>("addOrRemove", addOrRemove);
      program.template uniform<bool>("interOrIntra", true);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffersFuse(BufferFuseType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffersFuse(BufferFuseType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffersFuse(BufferFuseType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffersFuse(BufferFuseType::eCounts));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    if(addOrRemove) {
      auto &program = _programs(ProgramType::eFillNewNeighborsInter);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eNeighbors));

      for(uint i = 0; i < 2; ++i) {
        uint nSelected = selectionCounts[i];
        uint nSelectedOther = selectionCounts[1 - i];
        program.template uniform<uint>("nSelected", nSelected);
        program.template uniform<uint>("nSelectedOther", nSelectedOther);
        program.template uniform<uint>("offsetOther", (1 - i) * selectionCounts[0]);
        program.template uniform<float>("omissionRatio", std::clamp(1.f - (float) _params->k / nSelectedOther, 0.f, omissionRatioUpperBound(nSelectedOther)));
        program.template uniform<float>("omissionRatioOther", std::clamp(1.f - (float) _params->k / nSelected, 0.f, omissionRatioUpperBound(nSelected)));
        glDispatchCompute(ceilDiv(nSelected, 256u / 32u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      glAssert();
    }

    if(addOrRemove) {
      auto &program = _programs(ProgramType::eFillNewDistances);
      program.bind();

      program.template uniform<uint>("nSelectedAll", selectionCount);
      program.template uniform<uint>("nHighDims", _params->nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eDistancesL2));

      // Dispatch shader in batches of batchSize attriibutes
      uint batchSize = 5;
      for(uint b = 0; b * batchSize < _params->nHighDims; ++b) {
        program.template uniform<uint>("batchBegin", b * batchSize);
        program.template uniform<uint>("batchEnd", std::min((b+1) * batchSize, (uint) _params->nHighDims));
        glDispatchCompute(ceilDiv(selectionCount, 256u / 32u), 1, 1);
        glFinish();
        glAssert();
      }
    }

    if(addOrRemove) {
      auto& program = _programs(ProgramType::eFillNewSimilarities);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nSelectedAll", selectionCount);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("perplexity", _params->perplexity);
      program.template uniform<float>("epsilon", 1e-4);
      program.template uniform<float>("siphonRate", siphonRate);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eDistancesL2));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffersFuse(BufferFuseType::eSimilaritiesOriginal));

      // Dispatch shader
      glDispatchCompute(ceilDiv(selectionCount, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    std::swap(_buffers(BufferType::eLayout), _buffersFuse(BufferFuseType::eLayout));
    std::swap(_buffers(BufferType::eNeighbors), _buffersFuse(BufferFuseType::eNeighbors));
    std::swap(_buffers(BufferType::eSimilarities), _buffersFuse(BufferFuseType::eSimilarities));
    std::swap(_buffers(BufferType::eSimilaritiesOriginal), _buffersFuse(BufferFuseType::eSimilaritiesOriginal));
    std::swap(_buffers(BufferType::eDistancesL1), _buffersFuse(BufferFuseType::eDistancesL1));

    // Get new similarity per datapoint and store the factor with which to multiply the similarities for per-datapoint normalization
    // in eSimSumPerDatapointPre, cause I dont wanna create yet another buffer
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _params->n * sizeof(float), zeroesF.data(), 0);
    dh::util::BufferTools::instance().reducePerDatapoint<float>(_buffers(BufferType::eSimilarities), 0, _params->n, _buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    dh::util::BufferTools::instance().operate(1, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _params->n);
    weighSimilaritiesPerDatapoint(_buffersFuse(BufferFuseType::eSimSumPerDatapointPre));

    {
      auto &program = _programs(ProgramType::eSymmetrize);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilaritiesOriginal));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    glDeleteBuffers(_buffersFuse.size(), _buffersFuse.data());
    glAssert();
  }

  void Similarities::editLinksIntra(GLuint selectionBufferHandle, std::vector<uint> selectionCounts, bool addOrRemove, float siphonRate) {
    if(selectionCounts[0] == 0) { return; }

    // Get indices of selected datapoints into a buffer
    glCreateBuffers(_buffersFuse.size(), _buffersFuse.data());
    dh::util::BufferTools::instance().getIndices(selectionBufferHandle, _params->n, 1, _buffersFuse(BufferFuseType::eIndicesSelection));

    // eSizes contains every second element of eLayout
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSizes), _params->n * sizeof(uint), nullptr, 0);
    dh::util::BufferTools::instance().subsample(_buffers(BufferType::eLayout), _params->n, 2, 2, _buffersFuse(BufferFuseType::eSizes));

    // Get preexisting similarity both total and per datapoint
    std::vector<float> zeroes(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _params->n * sizeof(float), zeroes.data(), 0);
    dh::util::BufferTools::instance().reducePerDatapoint<float>(_buffers(BufferType::eSimilarities), 0, _params->n, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));

    // Get new nearest neighbor set sizes
    {
      auto& program = _programs(ProgramType::eUpdateSizesIntra);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eSizes));
      glAssert();

      program.template uniform<uint>("nSelected", selectionCounts[0]);
      program.template uniform<bool>("addOrRemove", addOrRemove);

      glDispatchCompute(ceilDiv(selectionCounts[0], 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    glNamedBufferStorage(_buffersFuse(BufferFuseType::eScan), _params->n * sizeof(uint), nullptr, 0);
    util::Scan(_buffersFuse(BufferFuseType::eSizes), _buffersFuse(BufferFuseType::eScan), _params->n).comp(); // Determine sizes of expanded neighborhoods in memory through prefix sum (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath
    glGetNamedBufferSubData(_buffersFuse(BufferFuseType::eScan), (_params->n - 1) * sizeof(uint), sizeof(uint), &_symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eLayout), _params->n * 2 * sizeof(uint), nullptr, 0);

    // Fill layout buffer
    {
      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersFuse(BufferFuseType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    // Create buffers for new links, will be swapped later
    std::vector<uint> zeroesU(_symmetricSize, 0);
    std::vector<float> zeroesF(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eNeighbors), _symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimilarities), _symmetricSize * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT); // Corresponding similarities
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimilaritiesOriginal), _symmetricSize * sizeof(float), nullptr, 0); // Corresponding similarities
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eDistancesL1), _symmetricSize * sizeof(float), zeroesF.data(), 0); // Corresponding distances
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eDistancesL2), _symmetricSize * sizeof(float), zeroesF.data(), 0); // Corresponding distances
    glNamedBufferStorage(_buffersFuse(BufferFuseType::eCounts), _params->n * sizeof(uint), zeroesU.data(), 0);

    {
      auto& program = _programs(ProgramType::eCopyOldStuff);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<bool>("addOrRemove", addOrRemove);
      program.template uniform<bool>("interOrIntra", false);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffersFuse(BufferFuseType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffersFuse(BufferFuseType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffersFuse(BufferFuseType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffersFuse(BufferFuseType::eCounts));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    if(addOrRemove) {
      auto &program = _programs(ProgramType::eFillNewNeighborsIntra);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eNeighbors));

      program.template uniform<uint>("nSelected", selectionCounts[0]);
      glDispatchCompute(ceilDiv(selectionCounts[0], 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    if(addOrRemove) {
      auto &program = _programs(ProgramType::eFillNewDistances);
      program.bind();

      program.template uniform<uint>("nSelectedAll", selectionCounts[0]);
      program.template uniform<uint>("nHighDims", _params->nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eDistancesL1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eDistancesL2));

      // Dispatch shader in batches of batchSize attriibutes
      uint batchSize = 5;
      for(uint b = 0; b * batchSize < _params->nHighDims; ++b) {
        program.template uniform<uint>("batchBegin", b * batchSize);
        program.template uniform<uint>("batchEnd", std::min((b+1) * batchSize, (uint) _params->nHighDims));
        glDispatchCompute(ceilDiv(selectionCounts[0], 256u / 32u), 1, 1);
        glFinish();
        glAssert();
      }
    }

    if(addOrRemove) {
      auto& program = _programs(ProgramType::eFillNewSimilarities);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nSelectedAll", selectionCounts[0]);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("perplexity", _params->perplexity);
      program.template uniform<float>("epsilon", 1e-4);
      program.template uniform<float>("siphonRate", siphonRate);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersFuse(BufferFuseType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersFuse(BufferFuseType::eIndicesSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersFuse(BufferFuseType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersFuse(BufferFuseType::eDistancesL2));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffersFuse(BufferFuseType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffersFuse(BufferFuseType::eSimilaritiesOriginal));

      // Dispatch shader
      glDispatchCompute(ceilDiv(selectionCounts[0], 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    std::swap(_buffers(BufferType::eLayout), _buffersFuse(BufferFuseType::eLayout));
    std::swap(_buffers(BufferType::eNeighbors), _buffersFuse(BufferFuseType::eNeighbors));
    std::swap(_buffers(BufferType::eSimilarities), _buffersFuse(BufferFuseType::eSimilarities));
    std::swap(_buffers(BufferType::eSimilaritiesOriginal), _buffersFuse(BufferFuseType::eSimilaritiesOriginal));
    std::swap(_buffers(BufferType::eDistancesL1), _buffersFuse(BufferFuseType::eDistancesL1));

    glNamedBufferStorage(_buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _params->n * sizeof(float), zeroes.data(), 0);
    dh::util::BufferTools::instance().reducePerDatapoint<float>(_buffers(BufferType::eSimilarities), 0, _params->n, _buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _buffers(BufferType::eLayout), _buffers(BufferType::eNeighbors));
    dh::util::BufferTools::instance().operate(1, _buffersFuse(BufferFuseType::eSimSumPerDatapointPre), _buffersFuse(BufferFuseType::eSimSumPerDatapointPst), _params->n);
    weighSimilaritiesPerDatapoint(_buffersFuse(BufferFuseType::eSimSumPerDatapointPre));

    {
      auto &program = _programs(ProgramType::eSymmetrize);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilaritiesOriginal));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    glDeleteBuffers(_buffersFuse.size(), _buffersFuse.data());
    glAssert();
  }

  void Similarities::reset() {
    glCopyNamedBufferSubData(_buffers(BufferType::eSimilaritiesOriginal), _buffers(BufferType::eSimilarities), 0, 0, _symmetricSize * sizeof(float));
  }

  void Similarities::assertSimilarities() {
    std::vector<uint> neig(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eNeighbors), 0, _symmetricSize * sizeof(uint), neig.data());
    std::vector<float> sims(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eSimilarities), 0, _symmetricSize * sizeof(float), sims.data());
    std::vector<uint> layo(_params->n * 2);
    glGetNamedBufferSubData(_buffers(BufferType::eLayout), 0, _params->n * 2 * sizeof(uint), layo.data());

    for(uint i = 0; i < _params->n; ++i) {
      std::set<uint> neighSet = std::set<uint>();

      for(uint ij = layo[i*2+0]; ij < layo[i*2+0] + layo[i*2+1]; ++ij) {
        uint j = neig[ij];

        if(neighSet.count(j) == 0) {
          neighSet.insert(j);
        } else {
          std::cerr << "\nDuplicate neighbor " << j << " for " << i << "!" << std::endl << std::flush;
          std::exit(1);
        }

        if(i == j) {
          std::cerr << "\nSelf-neighbor " << j << " for " << i << "!" << std::endl << std::flush;
          // std::exit(1);
        }

        bool foundSelf = false;
        for(uint ji = layo[j*2+0]; ji < layo[j*2+0] + layo[j*2+1]; ++ji) {
          if(neig[ji] == i) {
            foundSelf = true;
            if(sims[ij] != sims[ji]) {
              std::cerr << "\nMismatching similarities for " << i << " and " << j << "!" << std::endl << std::flush;
              std::exit(1);
            }
            break;
          }
        }
        if(!foundSelf) {
          std::cerr << "\nMissing neighbor " << i << " for " << j << "!" << std::endl << std::flush;
          std::exit(1);
        }
      }

      if(neighSet.size() != layo[i*2+1]) {
        std::cerr << "\nWrong number of neighbors for " << i << "!" << std::endl << std::flush;
        std::exit(1);
      }
    }
  }

  // Auxiliary function to create a window and display a scatterplot
  void Similarities::displayHistogram(std::vector<float> inter, std::vector<float> intra, bool relative) {
    std::vector<float> concat = intra;
    concat.insert(concat.end(), inter.begin(), inter.end());

    dh::util::GLWindowInfo info;
    {
      using namespace dh::util;
      info.title = "Graphs";
      info.width = 1500;
      info.height = 1000;
      info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                  | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                  | GLWindowInfo::bFloating;
    }
    dh::util::GLWindow window(info);

    ImGuiContext* ctxMain = ImGui::GetCurrentContext();
    IMGUI_CHECKVERSION();
    ImGuiContext* ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *) window.handle(), true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();

    window.setVsync(true);
    window.setVisible(true);

    ImPlotRange range = ImPlotRange(0, 1);
    if(inter.size() > 0 && intra.size() > 0) {
      auto [minInter, maxInter] = std::minmax_element(inter.begin(), inter.end());
      auto [minIntra, maxIntra] = std::minmax_element(intra.begin(), intra.end());
      ImPlotRange range = ImPlotRange(std::min(*minInter, *minIntra), std::max(*maxInter, *maxIntra));
    }

    int nBins = 100;

    while(window.canDisplay()) {
      window.processEvents();
      glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_NewFrame(); // Start new frame for IMGUI
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Appearing);
      ImGui::SetNextWindowSize(ImVec2(1500, 1500));
      ImGui::Begin("Graphs", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

      ImGui::SliderInt("No. of bins", &nBins, 1, 1000);

      if (ImPlot::BeginPlot("Inter/intra")) {
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
        if(inter.size() > 0) { ImPlot::PlotHistogram("Inter", inter.data(), inter.size(), nBins, 1.f, ImPlotRange(), relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(1), 0.5f);
        if(intra.size() > 0) { ImPlot::PlotHistogram("Intra", intra.data(), intra.size(), nBins, 1.f, ImPlotRange(), relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::EndPlot();
      }

      if (ImPlot::BeginPlot("Combined")) {
        if(concat.size() > 0) { ImPlot::PlotHistogram("All", concat.data(), concat.size(), nBins, 1.f, ImPlotRange(), relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::EndPlot();
      }

      ImGui::End();

      ImGui::Render(); // Finalize and render ImGui components
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      window.display();
    }
  }

  // Auxiliary function to create a window and display a scatterplot
  void Similarities::displayBarplot(std::vector<float> ys) {
    std::vector<float> xs(_params->nHighDims);
    std::iota(xs.begin(), xs.end(), 0); // Fills xs with 0..nHighDims-1

    dh::util::GLWindowInfo info;
    {
      using namespace dh::util;
      info.title = "Graphs";
      info.width = 1500;
      info.height = 1000;
      info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                  | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                  | GLWindowInfo::bFloating;
    }
    dh::util::GLWindow window(info);

    ImGuiContext* ctxMain = ImGui::GetCurrentContext();
    IMGUI_CHECKVERSION();
    ImGuiContext* ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *) window.handle(), true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();

    window.setVsync(true);
    window.setVisible(true);

    while(window.canDisplay()) {
      window.processEvents();
      glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_NewFrame(); // Start new frame for IMGUI
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Appearing);
      ImGui::SetNextWindowSize(ImVec2(1500, 1500));
      ImGui::Begin("Graphs", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

      if (ImPlot::BeginPlot("Barplot")) {
        ImPlot::PlotBars("Total distance", xs.data(), ys.data(), _params->nHighDims, 1.f);
        ImPlot::EndPlot();
      }

      ImGui::End();

      ImGui::Render(); // Finalize and render ImGui components
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      window.display();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    ImGui::SetCurrentContext(ctxMain);
    glAssert();
  }

} // dh::sne