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

#include <iostream>
#include <cuda_runtime.h>
#include <faiss/Clustering.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <glad/glad.h>
#include "dh/util/cu/KMeans.cuh"
#include "dh/util/cu/error.cuh"
#include "dh/util/gl/error.hpp"
#include "dh/util/io.hpp"

namespace dh::util {

  KMeans::KMeans() 
  : _isInit(false), _n(0), _d(0), _dataPtr(nullptr) {
    // ...
  }

  KMeans::KMeans(const float* dataPtr, uint n, uint d)
  : _isInit(false), _n(n), _d(d), _dataPtr(dataPtr) {

    _isInit = true;
  }

  // KMeans::KMeans(GLuint datasetBuffer, uint n, uint d, uint levels)
  // : _isInit(false), _n(n), _d(d), _levels(levels), _dataPtr(nullptr) {
    
  //   // Set up OpenGL-CUDA interoperability
  //   _interopBuffers(BufferType::eDataset) = CUGLInteropBuffer(datasetBuffer, CUGLInteropType::eNone);
  //   // _interopBuffers(BufferType::eCentroids) = CUGLInteropBuffer(centroidsBuffer, CUGLInteropType::eNone);

  //   _isInit = true;
  // }

  KMeans::~KMeans() {
    if (_isInit) {
      glDeleteBuffers(1, &_bufferCentroids);
    }
  }

  KMeans::KMeans(KMeans&& other) noexcept {
    swap(*this, other);
  }

  KMeans& KMeans::operator=(KMeans&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void KMeans::comp(uint nCentroids, bool spherical) {
    // Map interop buffers for access on CUDA side
    // _interopBuffers(BufferType::eCentroids).map();

    const float* dataPtr;
    if(_dataPtr) { dataPtr = _dataPtr; }
    else {
      _interopBuffers(BufferType::eDataset).map();
      dataPtr = (float*) _interopBuffers(BufferType::eDataset).cuHandle();
    }

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

    kMeans.train(_n, dataPtr, faissIndex);

    glCreateBuffers(1, &_bufferCentroids);
    glNamedBufferStorage(_bufferCentroids, nCentroids * _d * sizeof(float), kMeans.centroids.data(), 0);
    
    // Unmap interop buffers
    for (auto& buffer : _interopBuffers) {
      buffer.unmap();
    }

  }
} // dh::util