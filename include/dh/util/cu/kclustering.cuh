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

namespace dh::util {
  class KClustering {
  public:
    KClustering();
    KClustering(uint n, uint d, GLuint dataBufferHandle, GLuint indicesBufferHandle = 0);
    ~KClustering();

    // Copy constr/assignment is explicitly deleted (no copying handles)
    KClustering(const KClustering&) = delete;
    KClustering& operator=(const KClustering&) = delete;

    // Move constr/operator moves handles
    KClustering(KClustering&&) noexcept;
    KClustering& operator=(KClustering&&) noexcept;

    // Get buffer handles
    GLuint getDataBufferHandle() const { return _buffers(BufferType::eDatasetOut); }
    GLuint getIndicesBufferHandle() const { return _buffers(BufferType::eIndicesOut); }

    // Perform k-means/k-medoids computation, storing results in provided buffers
    void comp(uint nCentroids, bool medoids = false);

    bool isInit() const { return _isInit; }

  private:
    enum class BufferType {
      eDatasetIn,
      eIndicesIn,
      eDatasetOut,
      eIndicesOut,

      Length
    };

    enum class BufferTempType {
      eDistances,

      Length
    };

    bool _isInit;
    uint _n, _d;
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<BufferTempType, GLuint> _buffersTemp;

  public:
    // std::swap impl
    friend void swap(KClustering& a, KClustering& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._n, b._n);
      swap(a._d, b._d);
      swap(a._buffers, b._buffers);
      swap(a._buffersTemp, b._buffersTemp);
    }
  };
} // dh::util