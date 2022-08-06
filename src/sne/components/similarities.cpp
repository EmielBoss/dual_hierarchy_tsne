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
#include "dh/util/cu/inclusive_scan.cuh"
#include "dh/util/cu/knn.cuh"
#include <numeric> //
#include <fstream> //
#include <filesystem> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const float* dataPtr, Params params)
  : _isInit(false), _dataPtr(dataPtr), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/similarities.comp"));
      _programs(ProgramType::eExpandComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/expand.comp"));
      _programs(ProgramType::eLayoutComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/layout.comp"));
      _programs(ProgramType::eNeighborsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors.comp"));
      // _programs(ProgramType::eNeighborsSortComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors_sort.comp"));
      _programs(ProgramType::eWeightDistancesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weight_distances.comp"));

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Create and initialize buffers
    glCreateBuffers(_buffers.size(), _buffers.data());
    {
      const std::vector<uint> zeroes(_params.n * _params.k, 0);
      const std::vector<float> ones(_params.nHighDims, 1.0f);
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params.n * _params.nHighDims * sizeof(float), _dataPtr, 0); // Original dataset. TODO: make FAISS work from this instead of _dataPtr itself to avoid copying to GPU twice
      glNamedBufferStorage(_buffers(BufferType::eDistances), _params.n * _params.k * sizeof(float), nullptr, 0); // n * _params.k floats of neighbor distances; every _params.k'th element is 0
      glNamedBufferStorage(_buffers(BufferType::eKNNeighbors), _params.n * _params.k * sizeof(uint), nullptr, 0); // n * _params.k uints of neighbor indices (ranging from 0 to n-1); every _params.k'th element is vector index itself (so it's actually _params.k-1 NN)
      glNamedBufferStorage(_buffers(BufferType::eSizes), _params.n * sizeof(uint), zeroes.data(), 0); // n uints of (expanded) neighbor set sizes; every element is _params.k-1 plus its number of "unregistered neighbors" that have it as neighbor but that it doesn't reciprocate
      glNamedBufferStorage(_buffers(BufferType::eScan), _params.n * sizeof(uint), nullptr, 0); // Prefix sum/inclusive scan over expanded neighbor set sizes (eSizes)
      glNamedBufferStorage(_buffers(BufferType::eLayout), _params.n * 2 * sizeof(uint), nullptr, 0); // n structs of two uints; the first is its expanded neighbor set offset (eScan[i - 1]), the second is its expanded neighbor set size (eScan[i] - eScan[i - 1])
      glNamedBufferStorage(_buffers(BufferType::eAttributeWeights), _params.nHighDims * sizeof(float), ones.data(), GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
      glAssert();
    }

    _isInit = true;
    Logger::rest() << prefix << "Initialized";
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

  // Auxiliary function purely for debugging; will be removed
  template<typename T>
  void Similarities::writeBuffer(GLuint handle, uint n, uint d, std::string filename) {
    std::vector<T> buffer(n * d);
    glGetNamedBufferSubData(handle, 0, n * d * sizeof(T), buffer.data());
    std::ofstream file("../buffer_dumps/" + filename + ".txt");
    for(uint i = 0; i < n; i++) {
      for(uint j = 0; j < d; ++j) {
        T val = buffer[i * d + j];
        file << val << "|";
      }
      file << "\n";
    }
  }

  void Similarities::comp(GLuint selectedBufferHandle, std::set<uint> selectedAttributeIndices) {
    runtimeAssert(isInit(), "Similarities::comp() called without proper initialization");

    // Create and initialize temporary buffer objects
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    {
      const std::vector<uint> zeroes(_params.n * _params.k, 0);
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSimilarities), _params.n * _params.k * sizeof(float), zeroes.data(), 0); // n * _params.k floats of neighbor similarities; every _params.k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eCounts), _params.n * sizeof(uint), zeroes.data(), 0);
      glAssert();
    }
    
    // Progress bar for logging steps of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");
    progressBar.setPostfix("Performing KNN search");
    progressBar.setProgress(0.0f);
    
    // 1.
    // Compute approximate KNN of each point, delegated to FAISS
    // Produces a fixed number of neighbors
    if(selectedBufferHandle == 0) {
      util::KNN knn(
        _buffers(BufferType::eDataset),
        _buffers(BufferType::eDistances),
        _buffers(BufferType::eKNNeighbors),
        _params.n, _params.k, _params.nHighDims);
      knn.comp();
    }
    else {
      std::vector<uint> setvec(selectedAttributeIndices.begin(), selectedAttributeIndices.end());
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSelectedAttributeIndices), selectedAttributeIndices.size() * sizeof(uint), setvec.data(), 0);

      writeBuffer<uint>(_buffersTemp(BufferTempType::eSelectedAttributeIndices), selectedAttributeIndices.size(), 1, "selectedAttributeIndices");
      writeBuffer<float>(_buffers(BufferType::eAttributeWeights), _params.nHighDims, 1, "eAttributeWeights");

      auto& program = _programs(ProgramType::eWeightDistancesComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);
      program.template uniform<uint>("nHighDims", _params.nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectedBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSelectedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eAttributeWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eKNNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eDistances));

      // Dispatch shader in batches of 100 (selected) attriibutes
      for(uint b = 0; b * 100 < selectedAttributeIndices.size(); ++b) {
        uint begin = b*100;
        uint end = std::min((b+1) * 100, (uint) selectedAttributeIndices.size());
        uint size = selectedAttributeIndices.size();
        program.template uniform<uint>("batchBegin", b * 100);
        program.template uniform<uint>("batchEnd", std::min((b+1) * 100, (uint) selectedAttributeIndices.size()));
        glDispatchCompute(ceilDiv(_params.n * (_params.k-1), 256u), 1, 1);
        glFinish();
        glAssert();
      }
    }

    if(selectedBufferHandle == 0) { writeBuffer<float>(_buffers(BufferType::eDistances), _params.n, _params.k, "dist A"); }
    else { writeBuffer<float>(_buffers(BufferType::eDistances), _params.n, _params.k, "dist B"); }
    
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
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);
      program.template uniform<float>("perplexity", _params.perplexity);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eKNNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Symmetrizing KNN data");
    progressBar.setProgress(2.0f / 6.0f);

    // 3.
    // Expand KNN data so it becomes symmetric. That is, every neigbor referred by a point itself refers to that point as a neighbor.
    // Actually just fills eSizes, which, for each element, is _params.k-1+the number of "unregistered neighbors"; datapoints that have it as a neighbor but which it doesn't reciprocate
    if(selectedBufferHandle == 0) {
      auto& timer = _timers(TimerType::eExpandComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eExpandComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eKNNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Allocating buffers");
    progressBar.setProgress(3.0f / 6.0f);

    // 4.
    // Determine sizes of expanded neighborhoods in memory through prefix sum (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath

    uint symmetricSize;
    if(selectedBufferHandle == 0) {
      util::InclusiveScan scan(_buffers(BufferType::eSizes), _buffers(BufferType::eScan), _params.n);
      scan.comp();
      glGetNamedBufferSubData(_buffers(BufferType::eScan), (_params.n - 1) * sizeof(uint), sizeof(uint), &symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    
      // Initialize permanent buffer objects
      glNamedBufferStorage(_buffers(BufferType::eNeighbors), symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
      glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float), nullptr, 0); // Corresponding similarities
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Computing layout");
    progressBar.setProgress(4.0f / 6.0f);

    // 5.
    // Fill layout buffer
    if(selectedBufferHandle == 0) {
      auto& timer = _timers(TimerType::eLayoutComp);
      timer.tick();

      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Symmetrizing similarities");
    progressBar.setProgress(5.0f / 6.0f);

    // 6.
    // Generate expanded similarities and neighbor buffers, symmetrized and ready for use during the minimization
    {
      auto& timer = _timers(TimerType::eNeighborsComp);
      timer.tick();

      auto &program = _programs(ProgramType::eNeighborsComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eKNNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n * (_params.k-1), 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 7.
    // Sort neighbours within each KNN set (and corresponding similarities)
    // {
    //   auto &program = _programs(ProgramType::eNeighborsSortComp);
    //   program.bind();

    //   program.template uniform<uint>("nPoints", _params.n);

    //   // Set buffer bindings
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));

    //   // Dispatch shader
    //   glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
    //   glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //   glAssert();
    // }
    
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

} // dh::sne