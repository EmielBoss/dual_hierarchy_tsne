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

#pragma once

#include "dh/types.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/cu/interop.cuh"

namespace dh::util {
  class KClustering {
  public:
    KClustering();
    KClustering(const float* dataPtr, uint n, uint d);
    // KMeans(GLuint datasetBuffer, uint n, uint d);  // Faiss' kMeans doesn't take GPU buffers :(
    ~KClustering();

    // Copy constr/assignment is explicitly deleted (no copying handles)
    KClustering(const KClustering&) = delete;
    KClustering& operator=(const KClustering&) = delete;

    // Move constr/operator moves handles
    KClustering(KClustering&&) noexcept;
    KClustering& operator=(KClustering&&) noexcept;

    // Perform k-means/k-medoids computation, storing results in provided buffers
    void comp(uint nCentroids, bool medoids = false);

    bool isInit() const { return _isInit; }
    GLuint getResultsBuffer() const { return _bufferResults; }

  private:
    enum class BufferType {
      eDataset,

      Length
    };

    enum class BufferMedoidsType {
      eDataset,
      eDistances,
      eIndices,
      eIndicated,
      eThisNeedsToBeHereToAvoidACompileErrorForSomeReason,

      Length
    };

    bool _isInit;
    uint _n, _d;
    const float* _dataPtr;
    GLuint _bufferResults;
    EnumArray<BufferMedoidsType, GLuint> _buffersMedoids;
    EnumArray<BufferType, CUGLInteropBuffer> _interopBuffers;

  public:
    // std::swap impl
    friend void swap(KClustering& a, KClustering& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._n, b._n);
      swap(a._d, b._d);
      swap(a._dataPtr, b._dataPtr);
      swap(a._bufferResults, b._bufferResults);
      swap(a._interopBuffers, b._interopBuffers);
    }
  };
} // dh::util