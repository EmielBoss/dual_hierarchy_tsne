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

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");

  // Constants
  constexpr uint kMax = 192; // Don't exceeed this value for big vector datasets unless you have a lot of coffee and memory
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const std::vector<float>& data, Params params)
  : _isInit(false), _dataPtr(data.data()), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/similarities.comp"));
      _programs(ProgramType::eExpandComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/expand.comp"));
      _programs(ProgramType::eLayoutComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/layout.comp"));
      _programs(ProgramType::eNeighborsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors.comp"));

      _programs(ProgramType::eSelectionCountComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/selection_count.comp"));
      _programs(ProgramType::eUpdateSizesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/update_sizes.comp"));
      _programs(ProgramType::eIndicateSelectedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/indicate_selected.comp"));
      _programs(ProgramType::eSelectionIndicesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/selection_indices.comp"));
      _programs(ProgramType::eNeighborsUpdateComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/update_neighbours.comp"));
      _programs(ProgramType::eSimilaritiesUpdateComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/update_similarities.comp"));

      for (auto& program : _programs) {
        program.link();
      }
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

  void Similarities::comp() {
    runtimeAssert(isInit(), "Similarities::comp() called without proper initialization");

    // Actual k for KNN is limited to kMax, and is otherwise (3 * perplexity + 1)
    const uint k = std::min(kMax, 3 * static_cast<uint>(_params.perplexity) + 1);

    // Obtain buffer object handles
    glCreateBuffers(_buffers.size(), _buffers.data());
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    glAssert();

    // Initialize buffer objects. Comments state what data is supposed to go in the buffer (always row-wise, i.e. every vector in turn)
    {
      const std::vector<uint> zeroes(_params.n * k, 0);
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params.n * _params.nHighDims * sizeof(float), _dataPtr, 0); // Original dataset. TODO: make FAISS work from this instead of _dataPtr itself to avoid copying to GPU twice
      glNamedBufferStorage(_buffersTemp(BufferTempType::eNeighbors), _params.n * k * sizeof(uint), nullptr, 0); // n * k uints of neighbour indices (ranging from 0 to n-1); every k'th element is vector index itself (so it's actually k-1 NN)
      glNamedBufferStorage(_buffersTemp(BufferTempType::eDistances), _params.n * k * sizeof(float), nullptr, 0); // n * k floats of neighbour distances; every k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSimilarities), _params.n * k * sizeof(float), zeroes.data(), 0); // n * k floats of neighbour similarities; every k'th element is 0
      glNamedBufferStorage(_buffers(BufferType::eBetas), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eBetasFound), _params.n * sizeof(bool), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eV_jiSumBuffer), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSizes), _params.n * sizeof(uint), zeroes.data(), 0); // n uints of (expanded) neighbour set sizes; every element is k-1 plus its number of "unregistered neighbours" that have it as neighbour but that it doesn't reciprocate
      glNamedBufferStorage(_buffers(BufferType::eScan), _params.n * sizeof(uint), nullptr, 0); // Prefix sum/inclusive scan over expanded neighbor set sizes (eSizes)
      glNamedBufferStorage(_buffers(BufferType::eCounts), _params.n * sizeof(uint), zeroes.data(), 0);

      glNamedBufferStorage(_buffers(BufferType::eSelectionCount), 2 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectionCountReduce), 128 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelected), _params.n * sizeof(uint), nullptr, 0); // Bitfield indicating selection: 1 if selected, 0 if not selected (regardless of which selection)


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
    {
      util::KNN knn(
        _dataPtr,
        _buffersTemp(BufferTempType::eDistances),
        _buffersTemp(BufferTempType::eNeighbors),
        _params.n, k, _params.nHighDims);
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
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbours", k);
      program.template uniform<float>("perplexity", _params.perplexity);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eBetas));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eBetasFound));

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
    // Actually just fills eSizes, which, for each element, is k-1+the number of "unregistered neighbours"; datapoints that have it as a neighbour but which it doesn't reciprocate
    {
      auto& timer = _timers(TimerType::eExpandComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eExpandComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
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
    {
      util::InclusiveScan scan(_buffers(BufferType::eSizes), _buffers(BufferType::eScan), _params.n);
      scan.comp();
      glGetNamedBufferSubData(_buffers(BufferType::eScan), (_params.n - 1) * sizeof(uint), sizeof(uint), &symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    }
    _symmetricSize = symmetricSize; /////

    // Initialize permanent buffer objects
    glNamedBufferStorage(_buffers(BufferType::eLayout), _params.n * 2 * sizeof(uint), nullptr, 0); // n structs of two uints; the first is its expanded neighbor set offset (eScan[i - 1]), the second is its expanded neighbor set size (eScan[i] - eScan[i - 1])
    glNamedBufferStorage(_buffers(BufferType::eLayoutPrev), _params.n * 2 * sizeof(uint), nullptr, 0);
    glNamedBufferStorage(_buffers(BufferType::eNeighbors), symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbour set starts at eLayout[i].offset and contains eLayout[i].size neighbours, no longer including itself
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float), nullptr, 0); // Corresponding similarities
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
      program.template uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n * k, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
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

  // Update (i.e. add) neighbours to neighbour buffers and corresponding similarities
  void Similarities::update(GLuint selectionBuffer) {

    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    
    // 1.
    // Count number of selected datapoints per selection
    {
      glClearNamedBufferData(_buffers(BufferType::eSelectionCount), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); // Clear previous counts
      
      auto& program = _programs(ProgramType::eSelectionCountComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectionCountReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectionCount));
      glAssert();

      // Dispatch shader for selection i; i = {1, 2}
      for (uint i = 1; i < 3; ++i) {
        glClearNamedBufferData(_buffers(BufferType::eSelectionCountReduce), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        program.template uniform<uint>("selectionNumber", i);
        program.template uniform<uint>("iter", 0);
        glDispatchCompute(128, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        program.template uniform<uint>("iter", 1);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      glAssert();
    }

    uint selectionCounts[2];
    glGetNamedBufferSubData(_buffers(BufferType::eSelectionCount), 0, 2 * sizeof(uint), &selectionCounts);
    uint selectionCount = selectionCounts[0] + selectionCounts[1];

    //// DEBUGGIN
    std::vector<uint> sizesPrev(_params.n);
    glGetNamedBufferSubData(_buffers(BufferType::eSizes), 0, _params.n * sizeof(uint), sizesPrev.data());

    // 2.
    // Add selection counts (minus existing neighbours) to the NN size of each selected datapoint (eSizes)
    
    {
      auto &program = _programs(ProgramType::eUpdateSizesComp);
      std::cout << _programs(ProgramType::eUpdateSizesComp).getHandle() << "!!!";
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectionCount));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    //// DEBUGGING
    // std::vector<uint> selection(_params.n);
    // glGetNamedBufferSubData(selectionBuffer, 0, _params.n * sizeof(uint), selection.data());
    // std::vector<uint> layout(_params.n * 2);
    // glGetNamedBufferSubData(_buffers(BufferType::eLayout), 0, _params.n * 2 * sizeof(uint), layout.data());
    // std::vector<uint> sizes(_params.n);
    // glGetNamedBufferSubData(_buffers(BufferType::eSizes), 0, _params.n * sizeof(uint), sizes.data());
    

    //// UNCOMMENT FOR PROBLEM 2
    // std::vector<uint> neighbours(_symmetricSize);
    // glGetNamedBufferSubData(_buffersTemp(BufferTempType::eNeighbors), 0, _symmetricSize * sizeof(uint), neighbours.data());
    //// UNCOMMENT FOR PROBLEM 2


    // uint s1;
    // uint s2;
    // for(uint i = 0; i < _params.n; ++i) {
    //   if(selection[i] == 1) { s1 = i; }
    //   if(selection[i] == 2) { s2 = i; }
    // }

    // for(uint i = 0; i < _params.n; ++i) {
    //   uint s = selection[i];
    //   if(s == 0) { continue; }
    //   uint size = sizesPrev[i] + selectionCounts[s % 2];
    //   for(uint ij = layout[i * 2]; ij < layout[i * 2] + layout[i * 2 + 1]; ++ij) {
    //     uint j = neighbours[ij];
    //     if(selection[j] == (s % 2) + 1) { size--; }
    //   }
    //   if (size != sizes[i]) {
    //     for(uint ij = layout[i * 2]; ij < layout[i * 2] + layout[i * 2 + 1]; ++ij) {
    //       uint j = neighbours[ij];
    //       std::cout << "\n" << ij << ", " << j << ": " << selection[j];
    //     }
    //     std::cout << (size == sizes[i]) << "|";
    //   }
    // }

    // Prefix sum/inclusive sum over eSizes (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath
  
    uint neighbourTotal;
    {
      util::InclusiveScan scan(_buffers(BufferType::eSizes), _buffers(BufferType::eScan), _params.n);
      scan.comp();
      glGetNamedBufferSubData(_buffers(BufferType::eScan), (_params.n - 1) * sizeof(uint), sizeof(uint), &neighbourTotal); // Copy the last element (the total size) to host
    }

    // Update (actually: recompute) layout buffer

    std::swap(_buffers(BufferType::eLayout), _buffers(BufferType::eLayoutPrev));
    {
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
      glAssert();
    }

    // Getting a list of indices of selected datapoints, in order to avoid O(n^2)

    glClearNamedBufferData(_buffers(BufferType::eSelected), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    {
      auto& program = _programs(ProgramType::eIndicateSelectedComp); // Simply casts each 2 to 1 in eSelection
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelected));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    glNamedBufferStorage(_buffers(BufferType::eSelectedScan), _params.n * sizeof(uint), nullptr, 0); // The ordinal of the selected datapoint within the selection (and bullshit if unselected)
    util::InclusiveScan(_buffers(BufferType::eSelected), _buffers(BufferType::eSelectedScan), _params.n).comp();

    glNamedBufferStorage(_buffersTemp(BufferTempType::eSelectionIndices), selectionCount * sizeof(uint), nullptr, 0); // The vector indices of the selected datapoints

    {
      auto& program = _programs(ProgramType::eSelectionIndicesComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelected));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectedScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSelectionIndices));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    //// Update eNeighbours and eSimiliraties
    const std::vector<uint> zeroes(_params.n, 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eNeighbors), neighbourTotal * sizeof(uint), nullptr, 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eSimilarities), neighbourTotal * sizeof(float), nullptr, 0);
    glClearNamedBufferData(_buffers(BufferType::eCounts), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); // Clear previous counts
    glAssert();

    // Update eNeighbours

    {
      auto &program = _programs(ProgramType::eNeighborsUpdateComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("nSelected", selectionCount);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSelectionIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLayoutPrev));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffersTemp(BufferTempType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n * selectionCount, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    // // Calculate asymmetric similarities p_j|i

    {
      auto &program = _programs(ProgramType::eSimilaritiesUpdateComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("nHighDims", _params.nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, selectionBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSelectionIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLayoutPrev));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBetas));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eBetasFound));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::eV_jiSumBuffer));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffersTemp(BufferTempType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    std::swap(_buffers(BufferType::eLayout), _buffers(BufferType::eLayoutPrev));
    std::swap(_buffersTemp(BufferTempType::eNeighbors), _buffers(BufferType::eNeighbors));
    std::swap(_buffersTemp(BufferTempType::eSimilarities), _buffers(BufferType::eSimilarities));
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
  }

} // dh::sne