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

// #include <iostream>
#include <cuda_runtime.h>
#include <faiss/Clustering.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <glad/glad.h>
#include "dh/util/cu/kclustering.cuh"
#include "dh/util/cu/knn.cuh"
// #include "dh/util/cu/error.cuh"
// #include "dh/util/gl/error.hpp"
// #include "dh/util/io.hpp"

namespace dh::util {

  KClustering::KClustering() 
  : _isInit(false), _n(0), _d(0) {
    // ...
  }

  KClustering::KClustering(uint n, uint d, GLuint dataBufferHandle, GLuint indicesBufferHandle)
  : _isInit(false), _n(n), _d(d) {

    _buffers(BufferType::eDatasetIn) = dataBufferHandle;
    _buffers(BufferType::eIndicesIn) = indicesBufferHandle;
    glCreateBuffers(1, &_buffers(BufferType::eDatasetOut));
    glCreateBuffers(1, &_buffers(BufferType::eIndicesOut));
    _isInit = true;
  }

  KClustering::~KClustering() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    }
  }

  KClustering::KClustering(KClustering&& other) noexcept {
    swap(*this, other);
  }

  KClustering& KClustering::operator=(KClustering&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void KClustering::comp(uint nCentroids, bool medoids) {

    // Annoyingly, Faiss' clustering doesn't support GPU buffers, so we have to copy data over
    std::vector<float> dataIn(_n * _d);
    glGetNamedBufferSubData(_buffers(BufferType::eDatasetIn), 0, _n * _d * sizeof(float), dataIn.data());

    // Use a single GPU device. For now, just grab device 0 and pray
    faiss::gpu::StandardGpuResources faissResources;
    faiss::gpu::GpuIndexFlatConfig faissConfig;
    faissConfig.device = 0;
    faissConfig.useFloat16 = false;
    faiss::gpu::GpuIndexFlatL2 faissIndex(&faissResources, _d, faissConfig);

    faiss::ClusteringParameters faissClusteringParams;
    faissClusteringParams.niter = 30;
    faissClusteringParams.verbose = false;
    faissClusteringParams.seed = 0;
    faiss::Clustering kMeans(_d, nCentroids, faissClusteringParams);

    kMeans.train(_n, dataIn.data(), faissIndex);

    if(medoids) {
      glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
      glNamedBufferStorage(_buffers(BufferType::eIndicesOut), nCentroids * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffersTemp(BufferTempType::eDistances), nCentroids * _d * sizeof(float), nullptr, 0);
      util::KNN knn(_buffers(BufferType::eDatasetIn), _buffersTemp(BufferTempType::eDistances), _buffers(BufferType::eIndicesOut), _n, 1, _d);
      knn.comp(kMeans.centroids.data(), nCentroids);
      
      std::vector<uint> indices(nCentroids);
      glGetNamedBufferSubData(_buffers(BufferType::eIndicesOut), 0, nCentroids * sizeof(uint), indices.data());
      glNamedBufferStorage(_buffers(BufferType::eDatasetOut), nCentroids * _d * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
      for (uint i = 0; i < nCentroids; ++i) {
        glCopyNamedBufferSubData(_buffers(BufferType::eDatasetIn), _buffers(BufferType::eDatasetOut), indices[i] * _d * sizeof(float), i * _d * sizeof(float), _d * sizeof(float));
      }

      glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    } else {
      glNamedBufferStorage(_buffers(BufferType::eDatasetOut), nCentroids * _d * sizeof(float), kMeans.centroids.data(), 0);
    }

  }
} // dh::util