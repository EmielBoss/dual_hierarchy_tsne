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

#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include <resource_embed/resource_embed.hpp>
#include "dh/constants.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/util/cu/knn.cuh"
#include <faiss/VectorTransform.h>
#include <imgui.h>
#include <implot.h>
#include <numeric> //
#include <fstream> //
#include <filesystem> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Minimization]");

  // Params for field size
  constexpr uint fieldMinSize = 5;

  // Auxiliary function purely for debugging; will be removed
  template <uint D, uint DD>
  template<typename T>
  void Minimization<D, DD>::writeBuffer(GLuint handle, uint n, uint d, std::string filename) {
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

  template <uint D, uint DD>
  std::vector<float> Minimization<D, DD>::normalizeDataset(const float* dataPtr) {
    std::vector<float> mins(_params.nHighDims,  FLT_MAX);
    std::vector<float> maxs(_params.nHighDims, -FLT_MAX);
    for(uint i = 0; i < _params.n; ++i) {
      for(uint a = 0; a < _params.nHighDims; ++a) {
        float val = *(dataPtr + (i * _params.nHighDims) + a);
        if(val < mins[a]) { mins[a] = val; }
        if(val > maxs[a]) { maxs[a] = val; }
      }
    }

    std::vector<float> data(_params.n * _params.nHighDims);
    for(uint i = 0; i < _params.n; ++i) {
      for(uint a = 0; a < _params.nHighDims; ++a) {
        float val = *(dataPtr + (i * _params.nHighDims) + a);
        data[i * _params.nHighDims + a] = (val - mins[a]) / (maxs[a] - mins[a]);
        if(data[i * _params.nHighDims + a] != data[i * _params.nHighDims + a]) { data[i * _params.nHighDims + a] = 0.f; }
      }
    }
    return data;
  }

  template <uint D, uint DD>
  std::vector<float> Minimization<D, DD>::normalizeDatasetUniformScale(const float* dataPtr) {
    float min = FLT_MAX;
    float max = FLT_MIN;
    for(uint i = 0; i < _params.n * _params.nHighDims; ++i) {
      float val = *(dataPtr + i);
      if(val < min) { min = val; }
      if(val > max) { max = val; }
    }

    std::vector<float> data(_params.n * _params.nHighDims);
    for(uint i = 0; i < _params.n * _params.nHighDims; ++i) {
      float val = *(dataPtr + i);
      data[i] = (val - min) / (max - min);
      if(data[i] != data[i]) { data[i] = 0; }
    }
    return data;
  }

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization()
  : _isInit(false) {
    // ...
  }

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, Params params, std::vector<char> axisMapping)
  : _isInit(false), _loggedNewline(false), _similarities(similarities), _similaritiesBuffers(similarities->buffers()),
    _dataPtr(dataPtr), _selectionCounts(2, 0), _params(params), _axisMapping(axisMapping), _axisMappingPrev(axisMapping), _axisIndexPrev(-1),
    _draggedTexel(-1), _draggedTexelPrev(-1), _selectedDatapointPrev(0), _iteration(0) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eNeighborhoodPreservationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/neighborhood_preservation.comp"));
      _programs(ProgramType::eCountSelectedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/selection_count.comp"));
      _programs(ProgramType::eSelectionAverageComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/selection_average.comp"));
      _programs(ProgramType::eSelectionVarianceComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/selection_variance.comp"));
      _programs(ProgramType::eSelectionDifferenceComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/selection_difference.comp"));
      if constexpr (D == 2) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/gradients.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/centerEmbedding.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
      }
      if constexpr (DD == 2) {
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selection.comp"));
      } else if constexpr (DD == 3) {
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/selection.comp"));
      }
      if constexpr (D == 2 && DD == 3) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D_3DD/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D_3DD/translation.comp"));
      } else if constexpr (D == 2) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/translation.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/translation.comp"));
      }
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      const std::vector<vec> zerovecs(_params.n, vec(0));
      const std::vector<vec> unitvecs(_params.n, vec(1));
      const std::vector<uint> falses(_params.n, 0); // TODO: use bools instead of uints (but I can't seem to initialize buffers with bools; std::vector specializes <bool>)
      const std::vector<float> ones(_params.n, 1.0f);
      std::vector<float> zeros(_params.n, 0.0f);
      std::vector<uint> labeled(_params.n, 0);
      // for(uint i = 0; i < _params.n; ++i) { if (*(labelPtr + i) >= 0)  { labeled[i] = 1; } } // Use this when using regular labels (-1 = unlabeled, >0 = labeled)
      std::vector<uint> labeledIndices = {0, 1, 2, 3, 4, 5, 7, 13, 15, 17}; // Purely for development and demonstration on MNIST
      for(uint i = 0; i < labeledIndices.size(); ++i) { labeled[labeledIndices[i]] = 1; } // The first datapoints are "labeled"
      std::vector<float> data = normalizeDataset(dataPtr);

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params.n * _params.nHighDims * sizeof(float), data.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eLabels), _params.n * sizeof(int), labelPtr, 0);
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params.n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), unitvecs.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _params.n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), _params.n * sizeof(vec), zerovecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), _params.n * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborsEmb), _params.n * _params.k * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eDistancesEmb), _params.n * _params.k * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborhoodPreservation), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelection), _params.n * sizeof(uint), falses.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eSelectionCount), sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectionCountReduce), 128 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eTextureDataReduce), 128 * _params.nHighDims * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eFixed), _params.n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eTranslating), _params.n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are being translated
      glNamedBufferStorage(_buffers(BufferType::eWeights), _params.n * sizeof(float), ones.data(), 0); // The attractive force multiplier per datapoint
      glNamedBufferStorage(_buffers(BufferType::eLabeled), _params.n * sizeof(uint), labeled.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelative), _params.n * sizeof(vecc), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelativeBeforeTranslation), _params.n * sizeof(vec), nullptr, 0);
      glAssert();

      // Initialize everything required for textures
      if(_params.imageDataset) {
        _nTexels = _params.nHighDims / _params.imgDepth;
        std::vector<float> blacks(_nTexels * 4, 0.f);
        for(uint i = 0; i < _nTexels; ++i) { blacks[i * 4 + 3] = 1.f; }
        glCreateBuffers(_buffersTextureData.size(), _buffersTextureData.data());
        glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
        for(uint i = 0; i < _textures.size(); ++i) {
          glNamedBufferStorage(_buffersTextureData[i], 4 * _nTexels * sizeof(float), blacks.data(), GL_DYNAMIC_STORAGE_BIT);
          
          glTextureParameteri(_textures[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTextureParameteri(_textures[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
          glTextureStorage2D(_textures[i], 1, GL_RGBA8, _params.imgWidth, _params.imgHeight);
        }
        mirrorWeightsToOverlay();
      }
    }

    // Calculate nPCs number of principal components and stores the values in _pcs
    if(!_params.disablePCA) {
      faiss::PCAMatrix matrixPCA(_params.nHighDims, _params.nPCs);
      matrixPCA.train(_params.n, _dataPtr);
      _pcs = matrixPCA.apply(_params.n, _dataPtr);
    }

    initializeEmbeddingRandomly(_params.seed);

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Setup field subcomponent
    _field = Field<D>(buffers(), _params);

#ifdef DH_ENABLE_VIS_EMBEDDING
    // Setup input tasks
    if (auto& queue = vis::InputQueue::instance(); queue.isInit()) {
      _selectionInputTask = std::dynamic_pointer_cast<vis::SelectionInputTask>(vis::InputQueue::instance().find("SelectionInputTask"));
      if(DD == 3) { _trackballInputTask = std::dynamic_pointer_cast<vis::TrackballInputTask>(vis::InputQueue::instance().find("TrackballInputTask")); }
    }

    // Setup render tasks
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      std::string axistypesAbbr = "tpa-"; // Used to determine index of the selected axistype
      _axesRenderTask = queue.emplace(vis::AxesRenderTask<DD>(buffers(), _params, _axisMapping, axistypesAbbr.find(_axisMapping[2]), 1));
      _embeddingRenderTask = queue.emplace(vis::EmbeddingRenderTask<DD>(buffers(), _params, 0));
      _selectionRenderTask = queue.emplace(vis::SelectionRenderTask(_textures, _buffersTextureData, _similaritiesBuffers.attributeWeights, _params, 5, dataPtr));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    _isInit = true;
    glAssert();
  }

  // Generate randomized embedding data
  // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
  template <uint D, uint DD>
  void Minimization<D, DD>::initializeEmbeddingRandomly(int seed) {
    
    // Copy over embedding and fixed buffers to host
    std::vector<vec> embedding(_params.n);
    glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params.n * sizeof(vec), embedding.data());
    std::vector<uint> fixed(_params.n);
    glGetNamedBufferSubData(_buffers(BufferType::eFixed), 0, _params.n * sizeof(uint), fixed.data());

    // Seed the (bad) rng
    std::srand(seed);
    
    // Generate n random D-dimensional vectors
    for (uint i = 0; i < _params.n; ++i) {
      if(fixed[i]) { continue; }
      vec v;
      float r;

      do {
        r = 0.f;
        for (uint j = 0; j < D; ++j) {
          v[j] = 2.f * (static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.f)) - 1.f;
        }
        r = dot(v, v);
      } while (r > 1.f || r == 0.f);

      r = std::sqrt(-2.f * std::log(r) / r);
      embedding[i] = v * r * _params.rngRange;
    }

    // Copy to buffer
    glNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params.n * sizeof(vec), embedding.data());
    glAssert();
  }

  template <uint D, uint DD>
  Minimization<D, DD>::~Minimization() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteTextures(_buffersTextureData.size(), _buffersTextureData.data());
      glDeleteTextures(_textures.size(), _textures.data());
      _isInit = false;
    }
  }

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization(Minimization<D, DD>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D, uint DD>
  Minimization<D, DD>& Minimization<D, DD>::operator=(Minimization<D, DD>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // Restarts the minimization
  template <uint D, uint DD>
  void Minimization<D, DD>::restartMinimization() {
    if(_iteration < 100) { return; }
    if(_embeddingRenderTask->getReinitializeRandomly()) { initializeEmbeddingRandomly(_iteration); }
    else { initializeEmbeddingRandomly(_params.seed); }
    _iteration = 0;
    const std::vector<vec> zerovecs(_params.n, vec(0));
    glClearNamedBufferData(_buffers(BufferType::ePrevGradients), GL_R32F, GL_RED, GL_FLOAT, zerovecs.data());
  }

  // Configures the axes on request of change
  template <uint D, uint DD>
  void Minimization<D, DD>::reconfigureZAxis() {
    if constexpr (D == DD) { return; }

    std::vector<float> axisVals(_params.n);
    if(_axisMapping[2] == 'p') {
      for(uint i = 0; i < _params.n; ++i) { axisVals[i] = _pcs[i*_params.nPCs + _axisIndex]; }
    } else
    if(_axisMapping[2] == 'a') {
      for(uint i = 0; i < _params.n; ++i) { axisVals[i] = _dataPtr[i*_params.nHighDims + _axisIndex]; }
      if(_params.imageDataset) { setTexel(_axisIndex, {0.f, 1.f, 0.f, 1.f}); }
    }
    auto [minIt, maxIt] = std::minmax_element(axisVals.begin(), axisVals.end());
    float min = *minIt;
    float range = *maxIt - *minIt;
    float rangeInv = range > 0 ? 1 / range : 1;
    uint stride = (DD == 2) ? 2 : 4;
    for(uint i = 0; i < _params.n; ++i) {
      float valRel = (axisVals[i] - min) * rangeInv;
      glNamedBufferSubData(_buffers(BufferType::eEmbeddingRelative), (i*stride + D) * sizeof(float), sizeof(float), &valRel);
    }
    glAssert();
  }

  // Sets a specified texel in the overlay texture to the specified color
  template <uint D, uint DD>
  void Minimization<D, DD>::setTexel(int texelIndex, std::vector<float> color) {
    glNamedBufferSubData(_buffersTextureData(TextureType::eOverlay), texelIndex * 4 * sizeof(float), 4 * sizeof(float), color.data());
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::clearTextures() {
    for(uint i = 0; i < _buffersTextureData.size() - 1; ++i) {
      glClearNamedBufferData(_buffersTextureData[i], GL_R32F, GL_RED, GL_FLOAT, nullptr);
      float one = 1.f;
      for(uint t = 0; t < _nTexels; ++t) {
        glNamedBufferSubData(_buffersTextureData[i], (t * 4 + 3) * sizeof(float), sizeof(float), &one);
      }
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::brushTexels(uint centerTexelIndex, int radius, float weight) {
    // Create kernel
    std::vector<float> kernel(1, 1.f);
    for(int i = 1; i < 1 + radius * 2; ++i) {
      std::vector<float> kernelNext(i+1);
      kernelNext[0] = kernel[0];
      for(int j = 1; j < i; ++j) {
        kernelNext[j] = kernel[j-1] + kernel[j];
      }
      kernelNext[i] = kernel[i-1];
      kernel = kernelNext;
    }
    int max = *std::max_element(kernel.begin(), kernel.end());
    for(uint i = 0; i < kernel.size(); ++i) { kernel[i] /= max; } // Normalize kernel

    for(int i = -radius; i <= radius; ++i) {
      int x = centerTexelIndex / _params.imgWidth;
      if(x + i < 0 || x + i >= _params.imgHeight) { continue; }
      for(int j = -radius; j <= radius; ++j) {
        int y = centerTexelIndex % _params.imgWidth;
        if(y + j < 0 || y + j >= _params.imgWidth) { continue; }
        uint texelIndex = centerTexelIndex + i * _params.imgWidth + j;
        float texelWeightPrev;
        glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, texelIndex * sizeof(float), sizeof(float), &texelWeightPrev);
        float texelWeight = texelWeightPrev - (texelWeightPrev - weight) * kernel[i + radius] * kernel[j + radius];
        setTexelWeight(texelIndex, texelWeight);
      }
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::eraseTexels(uint centerTexelIndex, int radius) {
    for(int i = -radius; i <= radius; ++i) {
      int x = centerTexelIndex / _params.imgWidth;
      if(x + i < 0 || x + i >= _params.imgHeight) { continue; }
      for(int j = -radius; j <= radius; ++j) {
        int y = centerTexelIndex % _params.imgWidth;
        if(y + j < 0 || y + j >= _params.imgWidth) { continue; }
        uint texelIndex = centerTexelIndex + i * _params.imgWidth + j;
        setTexelWeight(texelIndex, 1.f);
      }
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::setTexelWeight(uint texelIndex, float weight) {
    for(uint i = 0; i < _params.imgDepth; ++i) {
      uint attr = texelIndex * _params.imgDepth + i;
      glNamedBufferSubData(_similaritiesBuffers.attributeWeights, attr * sizeof(float), sizeof(float), &weight);
      if(weight != 1.f) { _selectedAttributeIndices.insert(attr); } else { _selectedAttributeIndices.erase(attr); }
    }
    setTexel(texelIndex, {0.25f, 0.25f, 1.f, weight / _params.maxAttributeWeight / 1.5f});
  }

  template <uint D, uint DD>
  float Minimization<D, DD>::getTexelWeight(uint texelIndex) {
    std::vector<float> weights(_params.imgDepth);
    glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, texelIndex * _params.imgDepth * sizeof(float), _params.imgDepth * sizeof(float), weights.data());
    float weight = std::reduce(weights.begin(), weights.end());
    return weight / _params.imgDepth;
  }

  template <uint D, uint DD>
  float Minimization<D, DD>::getTexelValue(uint texelIndex, GLuint buffer) {
    std::vector<float> values(_params.imgDepth);
    glGetNamedBufferSubData(buffer, texelIndex * 4 * sizeof(float), _params.imgDepth * sizeof(float), values.data());
    float value = std::reduce(values.begin(), values.end());
    return value / _params.imgDepth;
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::mirrorWeightsToOverlay() {
    for(uint i = 0; i < _nTexels; ++i) {
      float weight = getTexelWeight(i) / _params.maxAttributeWeight;
      setTexel(i, {0.25f, 0.25f, 1.f, weight / 1.5f});
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::autoselectAttributes(uint textureType, float percentage) {
    std::vector<float> buffer(_nTexels * 4);
    glGetNamedBufferSubData(_buffersTextureData[textureType], 0, _nTexels * 4 * sizeof(float), buffer.data());
    std::vector<float> textureData(_nTexels, 0.f);
    for(uint i = 0; i < _nTexels; ++i) {
      for(uint c = 0; c < _params.imgDepth; ++c) {
        textureData[i] += buffer[i * 4 + c];
      }
      textureData[i] /= _params.imgDepth;
    }

    std::vector<size_t> indices(textureData.size());
    std::iota(indices.begin(), indices.end(), 0); // Fills indices with 0..nHighDims-1
    uint nSelected = _params.nHighDims * percentage;
    std::partial_sort(indices.begin(), indices.begin() + nSelected, indices.end(),
                      [&](size_t A, size_t B) {
                        return textureData[A] > textureData[B];
                      }); // Gives the nSelected indices of the largest values in textureData as nSelected first elements of indices
    
    for(uint i = 0; i < nSelected; ++i) {
      setTexelWeight(indices[i], _selectionRenderTask->getAttributeWeight());
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::invertAttributeWeights() {
    for(uint i = 0; i < _nTexels; ++i) {
      float weightTexel = getTexelWeight(i);
      float weightCurrent = _selectionRenderTask->getAttributeWeight();
      float weight = 1.f + weightCurrent - weightTexel;
      weight = std::clamp(weight, 0.f, _params.maxAttributeWeight);
      setTexelWeight(i, weight);
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::refineAttributeWeights() {
    std::vector<uint> attributeIndices(_selectedAttributeIndices.begin(), _selectedAttributeIndices.end());
    std::vector<uint> texelIndices;
    std::vector<float> values;
    GLuint buffer = _buffersTextureData[_selectionRenderTask->getTextureTabOpened()];
    for(uint i = 0; i < attributeIndices.size(); i = i + _params.imgDepth) {
      uint texelIndex = attributeIndices[i] / _params.imgDepth;
      float value = getTexelValue(texelIndex, buffer);
      values.push_back(value);
      texelIndices.push_back(texelIndex);
    }
    auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    float min = *minIt;
    float range = *maxIt - *minIt;

    for(uint i : texelIndices) {
      float value = getTexelValue(i, buffer);
      float ratio = (value - min) / range;

      float weightTexel = getTexelWeight(i);
      float weightCurrent = _selectionRenderTask->getAttributeWeight();
      float weight = 1.f - (1.f - weightCurrent) * ratio;
      weight = std::clamp(weight, 0.f, _params.maxAttributeWeight);
      setTexelWeight(i, weight);
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::comp() {
    while (_iteration < _params.iterations) {
      compIteration();
    }
  }

  // Core function handling everything that needs to happen each frame
  template <uint D, uint DD>
  bool Minimization<D, DD>::compIteration() {
    _input = _selectionInputTask->getInput();

    // Ugly synchronization work; I blame the existing software architecture
    {
      // Synchronizing color mapping between GUI and input
      _colorMappingPrev = _colorMapping;
      _colorMapping = _input.num;
      if(_colorMapping != _colorMappingPrev) { _embeddingRenderTask->setColorMapping(_colorMapping); }
      _colorMapping = _embeddingRenderTask->getColorMapping();
      if (_colorMapping != _colorMappingPrev) { _selectionInputTask->setNumPressed(_colorMapping); }

      // Synchronizing selection radius between GUI and input
      float selectionRadiusRelPrev = _selectionRadiusRel;
      _selectionRadiusRel = _selectionRenderTask->getSelectionRadiusRel();
      if (_selectionRadiusRel != selectionRadiusRelPrev) { _selectionInputTask->setMouseScroll(_selectionRadiusRel * 100); }
      _selectionRadiusRel = _input.mouseScroll / 100.f;
      if(_selectionRadiusRel != selectionRadiusRelPrev) { _selectionRenderTask->setSelectionRadiusRel(_selectionRadiusRel); }

      _selectionRenderTask->setMousePosScreen(_input.mousePosScreen); // Send screen position to GUI

      // Synchronize selection mode
      _selectOnlyLabeledPrev = _selectOnlyLabeled;
      _selectOnlyLabeled = _selectionRenderTask->getSelectionMode();
      _embeddingRenderTask->setSelectionMode(_selectOnlyLabeled);
    }

    // Get everything related with the cursor and selection brush
    if(_iteration < 1) { _window = util::GLWindow::currentWindow(); }
    glm::vec2 resolution = glm::vec2(_window->size());
    _model_view_2D = glm::translate(glm::vec3(-0.5f, -0.5f, -1.0f)); // TODO: get this directly from Rendered
    _proj_2D = glm::infinitePerspective(1.0f, resolution.x / resolution.y, 0.0001f); // TODO: get this directly from renderer
    if(DD == 3) {
      _model_view_3D = _trackballInputTask->matrix() * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f)); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)
      _proj_3D = glm::perspectiveFov(0.5f, resolution.x, resolution.y, 0.0001f, 1000.f); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)  
    }

    // Deselect
    if(_input.d) {
      std::fill(_selectionCounts.begin(), _selectionCounts.end(), 0);
      _selectionRenderTask->setSelectionCounts(_selectionCounts);
      _embeddingRenderTask->setWeighForces(true); // Use force weighting again; optional but may be convenient for the user
      glClearNamedBufferData(_buffers(BufferType::eSelection), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      if(_params.imageDataset) { clearTextures(); }
    }
    
    // Clear translations if not translating
    if(!_input.mouseRight) { glClearNamedBufferData(_buffers(BufferType::eTranslating), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); }
    
    // Free/unfix fixed datapoints
    if(_input.f) {
      const std::vector<float> ones(_params.n, 1.0f);
      glClearNamedBufferData(_buffers(BufferType::eWeights), GL_R32F, GL_RED, GL_FLOAT, ones.data());
      glClearNamedBufferData(_buffers(BufferType::eFixed), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    }

    if(_input.r) { restartMinimization(); } // Restart
    if(!_input.space) { compIterationMinimization(); } // Compute iteration, or pause if space is pressed
    if(_input.mouseLeft || _selectionRenderTask->getSelectAll()) { compIterationSelection(); } // Select
    if(_input.mouseRight || _mouseRightPrev) { compIterationTranslation(); } // Translate

    _mousePosClipPrev = _input.mousePosClip;
    _mouseLeftPrev = _input.mouseLeft;
    _mouseRightPrev = _input.mouseRight;

    if(_params.imageDataset) {
      // Draw dragselected attributes in texture
      _draggedTexel = _selectionRenderTask->getDraggedTexel();
      if(_draggedTexel >= 0 && _draggedTexel != _draggedTexelPrev) {
        if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          brushTexels(_draggedTexel, _selectionRenderTask->getTexelBrushRadius(), _selectionRenderTask->getAttributeWeight());
        } else
        if(ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
          eraseTexels(_draggedTexel, _selectionRenderTask->getTexelBrushRadius());
        }
        _draggedTexelPrev = _draggedTexel;
      }

      // Process attribute selection texture buttons (1 = Clear selection, 2 = Recomp distances, 3 = Recomp dataset, 4 = Recomp, 5 = Reset)
      _button = _selectionRenderTask->getButtonPressed();
      if(_button > 0 && _button != _buttonPrev) {
        if(_button == 1) { // Apply similarity weight
          _similarities->weighSimilarities(_selectionRenderTask->getSimilarityWeight(), _buffers(BufferType::eSelection));
        }
        if(_button == 10) { // Apply similarity weight to intersimilarities between selections
          _similarities->weighSimilarities(_selectionRenderTask->getSimilarityWeight(), _buffers(BufferType::eSelection), true);
        }
        if(_button == 15) { // Autoweigh/autoselect top % of attributes
          autoselectAttributes(_selectionRenderTask->getTextureTabOpened(), _selectionRenderTask->getAutoselectPercentage());
        }
        if(_button == 2) { // Recalc similarities
          _similarities->weighAttributes(_selectedAttributeIndices, _buffers(BufferType::eSelection), _selectionCounts[0], _buffers(BufferType::eLabels));
        }
        if(_button == 3) { // Reset similarities
          _similarities->reset();
        }
        if(_button == 4) { // Clear selection
          const std::vector<float> ones(_params.nHighDims, 1.0f);
          glClearNamedBufferData(_similaritiesBuffers.attributeWeights, GL_R32F, GL_RED, GL_FLOAT, ones.data());
          mirrorWeightsToOverlay();
        }
        if(_button == 5) { // Invert selection
          invertAttributeWeights();
        }
        if(_button == 6) { // Refine selection
          refineAttributeWeights();
        }
      }
      _similaritiesBuffers = _similarities->buffers(); // Refresh buffer handles, because some comps delete and recreate buffers
      _buttonPrev = _button;
    }

    // Select individual datapoints if int field input changed
    uint selectedDatapoint = (uint) _selectionRenderTask->getSelectedDatapoint();
    if(selectedDatapoint != _selectedDatapointPrev && selectedDatapoint < _params.n) {
      uint sn = _input.s + 1; // Selection number
      glNamedBufferSubData(_buffers(BufferType::eSelection), selectedDatapoint * sizeof(uint), sizeof(uint), &sn);
      compIterationSelection();
      _selectedDatapointPrev = selectedDatapoint;
    }

    // Reset some stuff upon axis change
    // _axisMapping = _axesRenderTask->getAxisMapping();
    // _axisIndex = _axesRenderTask->getSelectedIndex();
    // if(_axisIndex != _axisIndexPrev || _axisMapping != _axisMappingPrev) {
    //   if(_params.imageDataset) { # unset groene pixel # }
    //   reconfigureZAxis();
    //   _axisMappingPrev = _axisMapping;
    //   _axisIndexPrev = _axisIndex;
    // }

    // Copy texture data to textures
    if(_params.imageDataset && _iteration > 1) {
      for(uint i = 0; i < _textures.size(); ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTextureData[i]);
        glTextureSubImage2D(_textures[i], 0, 0, 0, _params.imgWidth, _params.imgHeight, GL_RGBA, GL_FLOAT, 0);
      }
    }

    // Reconstruct this Minimization if adding subtracting a t-SNE dimension
    // _axisMapping = _axesRenderTask->getAxisMapping();
    // _axisIndex = _axesRenderTask->getSelectedIndex();
    // if((_axisMapping[2] == 't' || _axisMappingPrev[2] == 't') && _axisMapping[2] != _axisMappingPrev[2]) {
    //   return true;
    // }

    glAssert();
    return false;
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationMinimization() {

    // 1.
    // Compute embedding bounds
    {
      auto& timer = _timers(TimerType::eBoundsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eBoundsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("padding", 0.0f);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eTranslating));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eBounds));

      // Dispatch shader
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Copy bounds back to host (hey look: an expensive thing I shouldn't be doing)
    _boundsPrev = _bounds;
    glGetNamedBufferSubData(_buffers(BufferType::eBounds), 0, sizeof(Bounds),  &_bounds);

    // 2.
    // Perform field approximation in subcomponent
    {
      // Determine field texture size by scaling bounds
      const vec range = _bounds.range();
      const float ratio = (D == 2) ? _params.fieldScaling2D : _params.fieldScaling3D;
      uvec size = dh::util::max(uvec(range * ratio), uvec(fieldMinSize));

      // Size becomes nearest larger power of two for field hierarchy
      size = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(size.x)) / glm::log(2.f))));

      // Delegate to subclass
      _field.comp(size, _iteration);
    }

    // 3.
    // Compute Z, ergo a reduction over q_{ij}
    {
      auto& timer = _timers(TimerType::eZComp);
      timer.tick();

      auto& program = _programs(ProgramType::eZComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eZReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));

      // Dispatch shader
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 4.
    // Compute attractive forces
    { 
      auto& timer = _timers(TimerType::eAttractiveComp);
      timer.tick();

      auto& program = _programs(ProgramType::eAttractiveComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPos", _params.n);
      program.template uniform<float>("invPos", 1.f / static_cast<float>(_params.n));
      program.template uniform<float>("weightFalloff", _embeddingRenderTask->getWeightFalloff());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.layout);  // n structs of two uints; the first is the offset into _similaritiesBuffers.neighbors where its kNN set starts, the second is the size of its kNN set
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.neighbors); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.similarities); // Corresponding similarities
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1); // One warp/sub21ee1group per datapoint
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Compute exaggeration factor
    float exaggeration = 1.0f;
    if (_iteration <= _params.removeExaggerationIter) {
      exaggeration = _params.exaggerationFactor;
    } else if (_iteration <= _params.removeExaggerationIter + _params.exponentialDecayIter) {
      float decay = 1.0f - static_cast<float>(_iteration - _params.removeExaggerationIter)
                         / static_cast<float>(_params.exponentialDecayIter);
      exaggeration = 1.0f + (_params.exaggerationFactor - 1.0f) * decay;
    }

    // 5.
    // Compute gradients
    {
      auto& timer = _timers(TimerType::eGradientsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eGradientsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttractive));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Precompute instead of doing it in shader N times
    const float iterMult = (static_cast<double>(_iteration) < _params.momentumSwitchIter) 
                         ? _params.momentum 
                         : _params.finalMomentum;

    // 6.
    // Update embedding
    {
      auto& timer = _timers(TimerType::eUpdateEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eUpdateEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("eta", _params.eta);
      program.template uniform<float>("minGain", _params.minimumGain);
      program.template uniform<float>("mult", 1.0);
      program.template uniform<float>("iterMult", iterMult);
      program.template uniform<bool>("translationFinished", !_input.mouseRight && _mouseRightPrev);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eGain));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 7.
    // Re-center embedding
    {
      const vec boundsCenter = _bounds.center();
      const vec boundsRange = _bounds.range();
      float scaling = 1.0f;
      if (exaggeration > 1.2f && boundsRange.y < 0.1f) {
        scaling = 0.1f / boundsRange.y;
      }
      
      auto& timer = _timers(TimerType::eCenterEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eCenterEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("scaling", scaling);
      program.template uniform<float, D>("center", boundsCenter);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 8.
    // Compute neighborhood preservation per datapoint

    if(_colorMapping == 2 && _colorMappingPrev != 2) {
      // Compute approximate KNN of each point in embedding, delegated to FAISS
      std::vector<vec> embedding(_params.n);
      glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params.n * sizeof(vec), embedding.data());
      std::vector<float> embeddingFiltered = dh::util::to_unaligned_vector<D, float>(embedding, _params.n);
      {
        util::KNN knn(
          embeddingFiltered.data(),
          _buffers(BufferType::eDistancesEmb),
          _buffers(BufferType::eNeighborsEmb),
          _params.n, _params.k, D);
        knn.comp();
      }

      {
        auto& program = _programs(ProgramType::eNeighborhoodPreservationComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);
        program.template uniform<uint>("kParam", _params.k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _similaritiesBuffers.layout);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similaritiesBuffers.neighbors);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighborsEmb));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighborhoodPreservation));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        glAssert();
      }
    }

    // Log progress; spawn progressbar on the current (new on first iter) line
    // reporting current iteration and size of field texture
    if (!_loggedNewline) {
      _loggedNewline = true;
    }
    if ((++_iteration % 100) == 0) {
      // Assemble string to print field's dimensions and memory usage
      std::stringstream fieldStr;
      {
        const uvec fieldSize = _field.size();
        fieldStr << fieldSize[0] << "x" << fieldSize[1];
        if constexpr (D == 3) {
          fieldStr << "x" << fieldSize[2];
        }
        fieldStr << " (" << (static_cast<float>(_field.memSize()) / 1'048'576.0f) << " mb)";
      }
      
      const std::string postfix = (_iteration < _params.iterations)
                                ? "iter: " + std::to_string(_iteration) + ", field: " + fieldStr.str()
                                : "Done!";
      util::ProgressBar progressBar(prefix + "Computing...", postfix);
      progressBar.setProgress(static_cast<float>(_iteration) / static_cast<float>(_params.iterations));
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationSelection() {
    uint si = _input.s; // Selection index
    uint sn = si + 1; // Selection number

    // 1.
    // Compute selection (or select all)
    if(_selectionRenderTask->getSelectAll()) {
      const std::vector<uint> data(_params.n, sn);
      glClearNamedBufferData(_buffers(BufferType::eSelection), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, data.data());
    } else
    {
      auto& program = _programs(ProgramType::eSelectionComp);
      program.bind();

      // Set uniform
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("selectionNumber", sn);
      program.template uniform<float, 2>("mousePosClip", _input.mousePosClip);
      program.template uniform<float>("selectionRadiusRel", _selectionRadiusRel);
      program.template uniform<float>("selectOnlyLabeled", _selectOnlyLabeled);
      program.template uniform<float, 4, 4>("model_view", DD == 2 ? _model_view_2D : _model_view_3D);
      program.template uniform<float, 4, 4>("proj", DD == 2 ? _proj_2D : _proj_3D);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLabeled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelection));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    // 2.
    // Count number of selected datapoints
    for (uint s = 0; s < 2; ++s) {
      auto& program = _programs(ProgramType::eCountSelectedComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("selectionNumber", s + 1);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectionCountReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectionCount));
      glAssert();

      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      uint selectionCountPrev = _selectionCounts[0];
      glGetNamedBufferSubData(_buffers(BufferType::eSelectionCount), 0, sizeof(uint), &_selectionCounts[s]);
      glAssert();

      _selectionRenderTask->setSelectionCounts(_selectionCounts);

      // Turn of force weighing if too many datapoints are selected at once, which is likely not what the user wants
      if(_selectionCounts[0] - selectionCountPrev > _params.n / 500) { _embeddingRenderTask->setWeighForces(false); }
    }

    // 3.
    // Calculate selection average and/or variance per attribute
    if(_params.imageDataset) {
      for(uint i = 0; i < 2; ++i) {
        auto& program = _programs(ProgramType::eSelectionAverageComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);
        program.template uniform<uint>("nSelected", _selectionCounts[i]);
        program.template uniform<uint>("imgSize", _params.nHighDims);
        program.template uniform<uint>("imgDepth", _params.imgDepth);
        program.template uniform<uint>("selectionNumber", i + 1);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelection));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eTextureDataReduce));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersTextureData[i * 2]);
        glAssert();

        program.template uniform<uint>("iter", 0);
        glDispatchCompute(128, _params.nHighDims, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        program.template uniform<uint>("iter", 1);
        glDispatchCompute(1,_params.nHighDims, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glAssert();
      }

      for(uint i = 0; i < 2; ++i) {
        auto& program = _programs(ProgramType::eSelectionVarianceComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);
        program.template uniform<uint>("nSelected", _selectionCounts[i]);
        program.template uniform<uint>("imgSize", _params.nHighDims);
        program.template uniform<uint>("imgDepth", _params.imgDepth);
        program.template uniform<uint>("selectionNumber", i + 1);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelection));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTextureData[i * 2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eTextureDataReduce));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersTextureData[i * 2 + 1]);
        glAssert();

        program.template uniform<uint>("iter", 0);
        glDispatchCompute(128, _params.nHighDims, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        program.template uniform<uint>("iter", 1);
        glDispatchCompute(1,_params.nHighDims, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glAssert();
      }

      for(uint i = 0; i < 2; ++i) {
        auto& program = _programs(ProgramType::eSelectionDifferenceComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nTexels", _nTexels);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTextureData[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTextureData[i+2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTextureData[i+4]);
        glAssert();

        glDispatchCompute(ceilDiv(_nTexels * 3, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glAssert();
      }
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationTranslation() {

    // Compute translation
    {
      glm::vec4 mousePosRel = glm::inverse(_proj_2D * _model_view_2D) * glm::vec4(_input.mousePosClip, 0.9998, 1);
      glm::vec4 mousePosRelPrev = glm::inverse(_proj_2D * _model_view_2D) * glm::vec4(_mousePosClipPrev, 0.9998, 1);
      vec shiftRel = mousePosRel - mousePosRelPrev;
      if constexpr (D == 3) {
        shiftRel = _trackballInputTask->changeOfBaseToCam() * shiftRel;
        shiftRel.y = -shiftRel.y;
      }

      auto& program = _programs(ProgramType::eTranslationComp);
      program.bind();

      // Set uniform
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float, D>("shiftRel", shiftRel);
      program.template uniform<bool>("translationStarted", !_mouseRightPrev);
      program.template uniform<bool>("translationFinished", !_input.mouseRight && _mouseRightPrev);
      program.template uniform<bool>("weighForces", _embeddingRenderTask->getWeighForces());
      program.template uniform<float>("weightFixed", _embeddingRenderTask->getWeightFixed());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eTranslating));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

  }

  // Template instantiations for 2/3 dimensions
  template class Minimization<2, 2>;
  template class Minimization<2, 3>;
  template class Minimization<3, 3>;
} // dh::sne