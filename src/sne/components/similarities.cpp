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
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/io.hpp"
#include "dh/util/cu/inclusive_scan.cuh"
#include "dh/util/cu/knn.cuh"
#include <typeinfo> //
#include <numeric> //
#include <imgui.h> //
#include <imgui_impl_glfw.h> //
#include <imgui_impl_opengl3.h> //
#include <implot.h> //

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
      _programs(ProgramType::eWeighSimilaritiesPerAttributeRatioComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_ratio.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeRangeComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_range.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeResembleComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr_resemble.comp"));
      _programs(ProgramType::eSubDistancesL1Comp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/subdistances.comp"));

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
      if(_params->uniformDims || _params->imageDataset) { dh::util::normalizeData(data, _params->n, _params->nHighDims); }
      else { dh::util::normalizeDataNonUniformDims(data, _params->n, _params->nHighDims); }

      glNamedBufferStorage(_buffers(BufferType::eDataset), _params->n * _params->nHighDims * sizeof(float), data.data(), 0);
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
      glNamedBufferStorage(_buffersTemp(BufferTempType::eDistances), _params->n * _params->k * sizeof(float), nullptr, 0); // n * k floats of neighbor distances; every k'th element is 0
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
        _buffersTemp(BufferTempType::eDistances),
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
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eDistances));
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

    {
      util::InclusiveScan scan(_buffersTemp(BufferTempType::eSizes), _buffersTemp(BufferTempType::eScan), _params->n);
      scan.comp();
      glGetNamedBufferSubData(_buffersTemp(BufferTempType::eScan), (_params->n - 1) * sizeof(uint), sizeof(uint), &_symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    }

    // Initialize permanent buffer objects
    std::vector<float> zeroes(_symmetricSize, 0.f);
    glNamedBufferStorage(_buffers(BufferType::eNeighbors), _symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), _symmetricSize * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT); // Corresponding similarities
    glNamedBufferStorage(_buffers(BufferType::eDistancesL1), _symmetricSize * sizeof(float), zeroes.data(), 0); // Corresponding distances
    glNamedBufferStorage(_buffers(BufferType::eNeighborsSelected), _symmetricSize * sizeof(uint), nullptr, 0); // Buffer used only by Minimization; creating it here because here we know the size
    glAssert();

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

    // 6.
    // Generate expanded similarities, neighbor and distances buffers, symmetrized and ready for use during the minimization
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
    // Calculating L1 distances
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
    glDeleteBuffers(1, &_buffers(BufferType::eNeighborsSelected));
    glCreateBuffers(1, &_buffers(BufferType::eNeighborsSelected));
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
    if(interOnly) { weight = std::pow(weight, 3); }

    auto &program = _programs(ProgramType::eWeighSimilaritiesComp);
    program.bind();

    program.template uniform<uint>("nPoints", _params->n);
    program.template uniform<float>("weight", weight);
    program.template uniform<bool>("weighAll", selectionBufferHandle == 0);
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

  void Similarities::weighSimilaritiesPerAttributeRatio(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, GLuint labelsBufferHandle) {
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

  void Similarities::weighSimilaritiesPerAttributeRange(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, GLuint labelsBufferHandle) {
    if(weightedAttributeIndices.size() == 0) { return; }

    // Create and initialize temp buffers
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> setvec(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), weightedAttributeIndices.size() * sizeof(uint), setvec.data(), 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eSubDistancesL1), _symmetricSize * sizeof(float), nullptr, 0);
    glClearNamedBufferData(_buffersTemp(BufferTempType::eSubDistancesL1), GL_R32F, GL_RED, GL_FLOAT, nullptr); // Initialize with all zeros

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

  void Similarities::weighSimilaritiesPerAttributeResemble(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, GLuint labelsBufferHandle, std::pair<uint, uint> snapslotHandles, uint nHighDims) {
    // Create and initialize temp buffers
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> attributeIndices;
    if(weightedAttributeIndices.size() > 0) {
      attributeIndices = std::vector<uint>(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    } else {
      attributeIndices = std::vector<uint>(nHighDims);
      std::iota(attributeIndices.begin(), attributeIndices.end(), 0);
      const std::vector<float> zeroes(_params->nHighDims, 0.f);
      glClearNamedBufferData(_buffers(BufferType::eAttributeWeights), GL_R32F, GL_RED, GL_FLOAT, zeroes.data());
    }
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), attributeIndices.size() * sizeof(uint), attributeIndices.data(), 0);

    // Weighting the similarities
    {
      auto &program = _programs(ProgramType::eWeighSimilaritiesPerAttributeResembleComp);
      program.bind();
      glAssert();

      float mult = (_params->nHighDims / weightedAttributeIndices.size()) / (5.f * (1.f - weightedAttributeIndices.size() / _params->nHighDims) + 1);
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      program.template uniform<uint>("nWeightedAttribs", attributeIndices.size());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, snapslotHandles.first);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, snapslotHandles.second);
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

  void Similarities::reset() {
    glCopyNamedBufferSubData(_buffers(BufferType::eSimilaritiesOriginal), _buffers(BufferType::eSimilarities), 0, 0, _symmetricSize * sizeof(float));
  }

} // dh::sne