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

#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/cu/kclustering.cuh"
#include "dh/vis/components/attribute_render_task.hpp"
#include "dh/util/io.hpp"

namespace dh::vis {
  
  AttributeRenderTask::AttributeRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  AttributeRenderTask::AttributeRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, sne::SimilaritiesBuffers similaritiesBuffers, GLuint colorBuffer, const int* labelPtr)
  : RenderTask(priority, "AttributeRenderTask"),
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _similaritiesBuffers(similaritiesBuffers),
    _bufferClassColors(colorBuffer),
    _params(params),
    _selectionCounts(2, 0),
    _draggedTexel(-1),
    _buttonPressed(0),
    _attributeWeight(0.f),
    _brushRadius(_params->imageDataset ? 1 : 0),
    _similarityWeight(2.f),
    _autoselectPercentage(0.025f),
    _tabIndex(0),
    _classIsSet(_params->nClasses, false),
    _classesSet(),
    _setChanged(false),
    _classButtonPressed(-1),
    _selectedDatapoint(-1),
    _separationMode(false),
    _vizAllPairs(false),
    _denominators(_textures.size(), 0),
    _archetypeClassSelected(0) {

    // Initialize shader program
    {
      _programs(ProgramType::ePairwiseAttrDiffsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/pairwise_attr_diffs.comp"));
      _programs(ProgramType::dGuessClasses).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/__guess_classes.comp"));
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffers
    {
      const std::vector<float> ones(_params->nHighDims, 1.0f);
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::ePairwiseAttrDists), _params->n * _params->nHighDims * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsSelectedPerDatapoint), _params->n * sizeof(int), nullptr, 0);
      glCreateBuffers(_buffersTextureData.size(), _buffersTextureData.data());
      for(uint i = 0; i < _buffersTextureData.size() - 1; ++i) {
        glNamedBufferStorage(_buffersTextureData[i], _params->nTexels * _params->imgDepth * sizeof(float), nullptr, 0);
      }
      glNamedBufferStorage(_buffersTextureData(TabType::eOverlay), _params->nTexels * 4 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
      mirrorWeightsToOverlay();
    }

    // Count classes
    _classCountsSelected = std::vector<uint>(_params->nClasses, 0);
    _classCounts = std::vector<uint>(_params->nClasses);
    for(uint i = 0; i < _params->nClasses; ++i) {
      // uint classCount = std::count(labelPtr, labelPtr + _params->n, i);
      uint classCount = dh::util::BufferTools::instance().reduce<int>(_minimizationBuffers.labels, 3, _params->n, 0, i);
      _classCounts[i] = classCount;
    }

    _classTextures = std::vector<GLuint>(_params->nClasses);
    // Create class textures for passing to _selectionRenderTask (doing this here because Minimization has the data and average() function)
    if(_params->imageDataset) {
      std::vector<GLuint> classTextureBuffers(_params->nClasses);
      glCreateBuffers(classTextureBuffers.size(), classTextureBuffers.data());
      glCreateTextures(GL_TEXTURE_2D, _classTextures.size(), _classTextures.data());
      for(uint i = 0; i < _params->nClasses; ++i) {
        glNamedBufferStorage(classTextureBuffers[i], _params->nTexels * _params->imgDepth * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glTextureParameteri(_classTextures[i], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(_classTextures[i], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GLenum formatInternal = _params->imgDepth == 1 ? GL_R8 : GL_RGB8;
        glTextureStorage2D(_classTextures[i], 1, formatInternal, _params->imgWidth, _params->imgHeight);
        
        dh::util::BufferTools::instance().averageTexturedata(_similaritiesBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.labels, _classCounts[i], classTextureBuffers[i], i);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, classTextureBuffers[i]);
        GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
        glTextureSubImage2D(_classTextures[i], 0, 0, 0, _params->imgWidth, _params->imgHeight, format, GL_FLOAT, 0);
      }
      glDeleteBuffers(classTextureBuffers.size(), classTextureBuffers.data());
    }

    // Create textures
    if(_params->imageDataset) {
      glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
      for(uint i = 0; i < _textures.size() - 1; ++i) {
        glTextureParameteri(_textures[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(_textures[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GLenum formatInternal = _params->imgDepth == 1 ? GL_R8 : GL_RGB8;
        glTextureStorage2D(_textures[i], 1, formatInternal, _params->imgWidth, _params->imgHeight);
        GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
        glClearTexImage(_textures[i], 0, format, GL_FLOAT, nullptr); // Avoids ugly texture initialization with random values
      }
      glTextureParameteri(_textures(TabType::eOverlay), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TabType::eOverlay), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(_textures(TabType::eOverlay), 1, GL_RGBA8, _params->imgWidth, _params->imgHeight);

      // Archetype textures and associates buffers and vectors
      _archetypeLabels = std::vector<uint>();
      _archetypeDatapointIndices = std::vector<uint>();
      _datapointArchetypeMapping = std::unordered_map<uint, uint>();
      _texturesSuggestions = std::vector<GLuint>();
    }

    _classNames = std::vector<std::string>(_params->nClasses, "");
    dh::util::readTxtClassNames(_params->datasetName + ".txt", _classNames, _params->nClasses);

    _classColors = std::vector<glm::vec3>(_params->nClasses);
    glGetNamedBufferSubData(_bufferClassColors, 0, _params->nClasses * sizeof(glm::vec3), _classColors.data());

    _isInit = true;
  }

  AttributeRenderTask::~AttributeRenderTask() {
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteBuffers(_buffersTextureData.size(), _buffersTextureData.data());
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteTextures(_texturesSuggestions.size(), _texturesSuggestions.data());
    glDeleteTextures(_classTextures.size(), _classTextures.data());
    glDeleteTextures(1, &_bufferClassColors);
  }

  AttributeRenderTask::AttributeRenderTask(AttributeRenderTask&& other) noexcept {
    swap(*this, other);
  }

  AttributeRenderTask& AttributeRenderTask::operator=(AttributeRenderTask&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // Sets a specified texel in the overlay texture to the specified color
  void AttributeRenderTask::setOverlayTexel(int texelIndex, std::vector<float> color) {
    glNamedBufferSubData(_buffersTextureData(TabType::eOverlay), texelIndex * 4 * sizeof(float), 4 * sizeof(float), color.data());
  }

  void AttributeRenderTask::brushTexels(uint centerTexelIndex, int radius, float weight) {
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
      int x = centerTexelIndex / _params->imgWidth;
      if(x + i < 0 || x + i >= _params->imgHeight) { continue; }
      for(int j = -radius; j <= radius; ++j) {
        int y = centerTexelIndex % _params->imgWidth;
        if(y + j < 0 || y + j >= _params->imgWidth) { continue; }
        uint texelIndex = centerTexelIndex + i * _params->imgWidth + j;
        float texelWeightPrev;
        glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, texelIndex * sizeof(float), sizeof(float), &texelWeightPrev);
        float texelWeight = texelWeightPrev - (texelWeightPrev - weight) * kernel[i + radius] * kernel[j + radius];
        setTexelWeight(texelIndex, texelWeight);
      }
    }
    copyTextureDataToTextures();
  }

  void AttributeRenderTask::eraseTexels(uint centerTexelIndex, int radius) {
    for(int i = -radius; i <= radius; ++i) {
      int x = centerTexelIndex / _params->imgWidth;
      if(x + i < 0 || x + i >= _params->imgHeight) { continue; }
      for(int j = -radius; j <= radius; ++j) {
        int y = centerTexelIndex % _params->imgWidth;
        if(y + j < 0 || y + j >= _params->imgWidth) { continue; }
        uint texelIndex = centerTexelIndex + i * _params->imgWidth + j;
        setTexelWeight(texelIndex, 1.f);
      }
    }
    copyTextureDataToTextures();
  }

  void AttributeRenderTask::setTexelWeight(uint texelIndex, float weight) {
    for(uint i = 0; i < _params->imgDepth; ++i) {
      uint attr = texelIndex * _params->imgDepth + i;
      glNamedBufferSubData(_similaritiesBuffers.attributeWeights, attr * sizeof(float), sizeof(float), &weight);
      if(weight != 1.f) { _weightedAttributeIndices.insert(attr); } else { _weightedAttributeIndices.erase(attr); }
    }
    setOverlayTexel(texelIndex, {0.25f, 0.25f, 1.f, weight / _params->maxAttributeWeight / 1.5f});
  }

  float AttributeRenderTask::getTexelWeight(uint texelIndex) {
    std::vector<float> weights(_params->imgDepth);
    glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, texelIndex * _params->imgDepth * sizeof(float), _params->imgDepth * sizeof(float), weights.data());
    float weight = std::reduce(weights.begin(), weights.end());
    return weight / _params->imgDepth;
  }

  float AttributeRenderTask::getTexelValue(uint texelIndex, GLuint buffer) {
    std::vector<float> values(_params->imgDepth);
    glGetNamedBufferSubData(buffer, texelIndex * _params->imgDepth * sizeof(float), _params->imgDepth * sizeof(float), values.data());
    float value = std::reduce(values.begin(), values.end());
    return value / _params->imgDepth;
  }

  void AttributeRenderTask::mirrorWeightsToOverlay() {
    for(uint i = 0; i < _params->nTexels; ++i) {
      float weight = getTexelWeight(i) / _params->maxAttributeWeight;
      setOverlayTexel(i, {0.25f, 0.25f, 1.f, weight / 1.5f});
    }
  }

  void AttributeRenderTask::autoweighAttributes(uint textureType, float percentage) {
    std::vector<float> textureBuffer(_params->nTexels * _params->imgDepth);
    glGetNamedBufferSubData(_buffersTextureData[textureType], 0, _params->nTexels * _params->imgDepth * sizeof(float), textureBuffer.data());
    std::vector<float> textureData(_params->nTexels, 0.f);
    for(uint i = 0; i < _params->nTexels; ++i) {
      for(uint c = 0; c < _params->imgDepth; ++c) {
        textureData[i] += textureBuffer[i * _params->imgDepth + c];
      }
      textureData[i] /= _params->imgDepth;
    }

    std::vector<size_t> indices(textureData.size());
    std::iota(indices.begin(), indices.end(), 0); // Fills indices with 0..nHighDims-1
    uint nSelected = _params->nTexels * percentage;
    std::partial_sort(indices.begin(), indices.begin() + nSelected, indices.end(),
                      [&](size_t A, size_t B) {
                        return textureData[A] > textureData[B];
                      }); // Gives the nSelected indices of the largest values in textureData as nSelected first elements of indices
    
    for(uint i = 0; i < nSelected; ++i) {
      setTexelWeight(indices[i], _attributeWeight);
    }

    copyTextureDataToTextures();
  }

  void AttributeRenderTask::clearAttributeWeights() {
    const std::vector<float> ones(_params->nHighDims, 1.0f);
    glClearNamedBufferData(_similaritiesBuffers.attributeWeights, GL_R32F, GL_RED, GL_FLOAT, ones.data());
    mirrorWeightsToOverlay();
    _weightedAttributeIndices = std::set<uint>();
    if(_params->imageDataset) { copyTextureDataToTextures(); }
  }

  void AttributeRenderTask::invertAttributeWeights() {
    for(uint i = 0; i < _params->nTexels; ++i) {
      float weightTexel = getTexelWeight(i);
      float weightCurrent = _attributeWeight;
      float weight = 1.f + weightCurrent - weightTexel;
      weight = std::clamp(weight, 0.f, _params->maxAttributeWeight);
      setTexelWeight(i, weight);
    }
    if(_params->imageDataset) { copyTextureDataToTextures(); }
  }

  void AttributeRenderTask::refineAttributeWeights(uint textureType) {
    float min = 0.f;
    float max = 1.f;
    for(uint i = 0; i < _params->nTexels; i = ++i) {
      if(_weightedAttributeIndices.find(i) == _weightedAttributeIndices.end()) { continue; } // Attribute isn't weighted

      float value = getTexelValue(i, _buffersTextureData[textureType]);
      float ratio = (value - min) / (max - min);

      float weightOld = getTexelWeight(i);
      float weightCurrent = _attributeWeight;
      float weightNew = 1.f - (1.f - weightCurrent) * ratio;
      float weight = weightOld * 1.f + (1.f - weightOld) * weightNew;
      if(weight > 1.f - 0.025 && weight < 1.f + 0.025) { weight = 1.f; }
      if(weight < 0.025) { weight = 0.f; }
      setTexelWeight(i, weight);
    }
    if(_params->imageDataset) { copyTextureDataToTextures(); }
  }

  float AttributeRenderTask::sumWeightedAttributeValues(uint index) {
    std::vector<float> weights(_params->nHighDims);
    glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, 0, _params->nHighDims * sizeof(float), weights.data());
    std::vector<float> values(_params->nHighDims);
    glGetNamedBufferSubData(_buffersTextureData[index], 0, _params->nHighDims * sizeof(float), values.data());

    float sumValues = 0.f;
    float sumWeights = 0.f;
    for (uint i : _weightedAttributeIndices) {
      sumValues += values[i] * (1 - weights[i]);
      sumWeights += 1 - weights[i];
    }

    return sumValues / sumWeights;
  }

  void AttributeRenderTask::copyTextureDataToTextures() {
    if(!_params->imageDataset) { return; }

    // Copy texture data to textures
    for(uint i = 0; i < _textures.size() - 1; ++i) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTextureData[i]);
      GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
      glTextureSubImage2D(_textures[i], 0, 0, 0, _params->imgWidth, _params->imgHeight, format, GL_FLOAT, 0);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTextureData(TabType::eOverlay));
    glTextureSubImage2D(_textures(TabType::eOverlay), 0, 0, 0, _params->imgWidth, _params->imgHeight, GL_RGBA, GL_FLOAT, 0);
    glAssert();
  }

  void AttributeRenderTask::addArchetype(uint archetypeDatapointIndex, uint archetypeClass) {
    _archetypeLabels.push_back(archetypeClass);
    _archetypeDatapointIndices.push_back(archetypeDatapointIndex);
    _datapointArchetypeMapping[archetypeDatapointIndex] = _archetypeLabels.size() - 1;
    glAssert();
  }

  void AttributeRenderTask::removeArchetype(uint datapointIndex) {
    uint archetypeIndex = _datapointArchetypeMapping[datapointIndex];
    _archetypeLabels.erase(_archetypeLabels.begin() + archetypeIndex);
    _archetypeDatapointIndices.erase(_archetypeDatapointIndices.begin() + archetypeIndex);
    _datapointArchetypeMapping.erase(datapointIndex);
    for(uint i = 0; i < _archetypeDatapointIndices.size(); ++i) {
      _datapointArchetypeMapping[_archetypeDatapointIndices[i]] = i;
    }
  }

  void AttributeRenderTask::clearArchetypes() {
    _archetypeLabels.clear();
    _archetypeDatapointIndices.clear();
    _datapointArchetypeMapping.clear();
  }

  void AttributeRenderTask::updateSuggestions() {
    uint nSuggestionsNew = std::pow(2, _suggestionLevel);
    uint nSuggestionsPrev = _indicesSuggestions.size();

    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    dh::util::BufferTools::instance().remove<float>(_similaritiesBuffers.dataset, _params->n, _params->nHighDims, _minimizationBuffers.selection, _buffersTemp(BufferTempType::eDatasetSelection));

    std::vector<int> indices(_params->n);
    std::iota(indices.begin(), indices.end(), 0);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eIndicesSelection), _params->n * sizeof(int), indices.data(), 0);
    dh::util::BufferTools::instance().remove<int>(_buffersTemp(BufferTempType::eIndicesSelection), _params->n, 1, _minimizationBuffers.selection);

    util::KClustering kClustering(_selectionCounts[0] + _selectionCounts[1], _params->nHighDims, _buffersTemp(BufferTempType::eDatasetSelection));
    kClustering.comp(nSuggestionsNew, true);

    _texturesSuggestions.resize(nSuggestionsPrev + nSuggestionsNew);
    glCreateTextures(GL_TEXTURE_2D, nSuggestionsNew, &_texturesSuggestions[nSuggestionsPrev]);
    glNamedBufferStorage(_buffersTemp(BufferTempType::eArchetypeTemp), _params->nHighDims * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
    for(uint i = 0; i < nSuggestionsNew; ++i) {
      glCopyNamedBufferSubData(kClustering.getDataBufferHandle(), _buffersTemp(BufferTempType::eArchetypeTemp), i * _params->nHighDims * sizeof(float), 0, _params->nHighDims * sizeof(float));
      glTextureParameteri(_texturesSuggestions[nSuggestionsPrev + i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_texturesSuggestions[nSuggestionsPrev + i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      GLenum formatInternal = _params->imgDepth == 1 ? GL_R8 : GL_RGB8;
      glTextureStorage2D(_texturesSuggestions[nSuggestionsPrev + i], 1, formatInternal, _params->imgWidth, _params->imgHeight);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTemp(BufferTempType::eArchetypeTemp));
      GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
      glTextureSubImage2D(_texturesSuggestions[nSuggestionsPrev + i], 0, 0, 0, _params->imgWidth, _params->imgHeight, format, GL_FLOAT, 0);
    }

    // Map indicesOut to IndicesIn, because Faiss' KNN GPU implementation for some reason breaks with custom indices
    // See https://github.com/facebookresearch/faiss/issues/3053
    // I'd rather do this in KCLustering.comp() but I can't get that to work
    GLuint indicesOutHandle = kClustering.getIndicesBufferHandle();
    dh::util::BufferTools::instance().index(indicesOutHandle, _buffersTemp(BufferTempType::eIndicesSelection), nSuggestionsNew);
    std::vector<uint> indicesOut(nSuggestionsNew);
    glGetNamedBufferSubData(indicesOutHandle, 0, nSuggestionsNew * sizeof(uint), indicesOut.data());
    _indicesSuggestions.insert(_indicesSuggestions.end(), indicesOut.begin(), indicesOut.end());

    // Remove duplicate suggestions
    for(uint i = nSuggestionsPrev; i < nSuggestionsPrev + nSuggestionsNew; ++i) {
      for(uint j = 0; j < nSuggestionsPrev; ++j) {
        if(_indicesSuggestions[i] == _indicesSuggestions[j]) {
          glDeleteTextures(1, &_texturesSuggestions[i]);
          _indicesSuggestions.erase(_indicesSuggestions.begin() + i);
          _texturesSuggestions.erase(_texturesSuggestions.begin() + i);
          --i;
          --nSuggestionsNew;
          break;
        }
      }
    }

    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    glAssert();
    _suggestionLevel++;
  }

  void AttributeRenderTask::updateDatapointIndicesForArchetypes() {
    GLuint tempScanBuffer;
    glCreateBuffers(1, &tempScanBuffer);
    glNamedBufferStorage(tempScanBuffer, _params->n * sizeof(int), nullptr, 0);
    util::Scan(_minimizationBuffers.selection, tempScanBuffer, _params->n).comp(false);

    std::vector<int> scan(_params->n);
    glGetNamedBufferSubData(tempScanBuffer, 0, _params->n * sizeof(int), scan.data());
    std::vector<int> selection(_params->n);
    glGetNamedBufferSubData(_minimizationBuffers.selection, 0, _params->n * sizeof(int), selection.data());

    for(uint i = 0; i < _archetypeDatapointIndices.size(); ++i) {
      uint datapointIndex = _archetypeDatapointIndices[i];
      if(selection[datapointIndex] == 0) {
        removeArchetype(datapointIndex);
        --i;
      } else {
        int nCumulativeDatapointsRemoved = datapointIndex - scan[datapointIndex];
        _archetypeDatapointIndices[i] -= nCumulativeDatapointsRemoved;
        _datapointArchetypeMapping.erase(datapointIndex);
        _datapointArchetypeMapping[_archetypeDatapointIndices[i]] = i;
      }
    }

    glDeleteBuffers(1, &tempScanBuffer);
    glAssert();
  }

  void AttributeRenderTask::render(glm::mat4 model_view, glm::mat4 proj) {
    if (!enabled) {
      return;
    }
    glAssert();
  }

  void AttributeRenderTask::drawImGuiComponent() {
    _buttonPressed = 0;
    int tabUpper = -1;
    if (ImGui::CollapsingHeader("Attribute render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::BeginTabBar("Upper tabs")) {

        if (ImGui::BeginTabItem("Selection pri")) {
          tabUpper = TabUpperType::eSelectionPrimary;
          ImGui::EndTabItem();
        }

        if(_selectionCounts[1] > 0) {
          if (ImGui::BeginTabItem("Selection sec")) {
            tabUpper = TabUpperType::eSelectionSecondary;
            ImGui::EndTabItem();
          }

          // if (ImGui::BeginTabItem("Selection diff")) {
          //   tabUpper = TabUpperType::eSelectionDifference;
          //   ImGui::EndTabItem();
          // }
        }

        if (ImGui::BeginTabItem("Pairwise")) {
          tabUpper = TabUpperType::ePairwise;
          ImGui::EndTabItem();
        }

        if(_params->imageDataset) {
          if (ImGui::BeginTabItem("Suggestions")) {
            tabUpper = TabUpperType::eSuggestions;
            ImGui::EndTabItem();
          }
        }

        ImGui::EndTabBar();
      }

      _draggedTexel = -1;

      if (tabUpper < TabUpperType::eSuggestions && ImGui::BeginTabBar("Lower tabs")) {
        if(tabUpper == TabUpperType::eSelectionPrimary) {
          drawImGuiTab(TabType::eAveragePrimary, "Average");
          drawImGuiTab(TabType::eVariancePrimary, "Variance");
        } else
        if(tabUpper == TabUpperType::eSelectionSecondary) {
          drawImGuiTab(TabType::eAverageSecondary, "Average");
          drawImGuiTab(TabType::eVarianceSecondary, "Variance");
        } else
        if(tabUpper == TabUpperType::eSelectionDifference) {
          drawImGuiTab(TabType::eAverageDifference, "Average");
          drawImGuiTab(TabType::eVarianceDifference, "Variance");
        } else
        if(tabUpper == TabUpperType::ePairwise) {
          drawImGuiTab(TabType::ePairwiseDiffsAll, "All");
          if(_classesSet.size() == 2) {
            drawImGuiTab(TabType::ePairwiseDiffsInterclass, "Interclass");
            drawImGuiTab(TabType::ePairwiseDiffsIntraclass, "Intraclass");
          }
          if(_selectionCounts[0] > 0 && _selectionCounts[1] > 0) {
            drawImGuiTab(TabType::ePairwiseDiffsInterselection, "Interselection");
            drawImGuiTab(TabType::ePairwiseDiffsIntraselection, "Intraselection");
          }
        }
        ImGui::EndTabBar();
      }

      if(tabUpper == TabUpperType::eSuggestions) {
        _tabIndex = -1;
        drawImGuiSuggestor();
      }
    }

    // Force selection re-evaluation when switching to the pairwise attr diff tab
    if(_tabIndex != _tabIndexPrev || _setChanged) {
      if(tabUpper == TabUpperType::ePairwise) {
        updateVisualizations(_selectionCounts);
      } else
      if(tabUpper == TabUpperType::eSuggestions) {
        clearSuggestions();
        if(_selectionCounts[0] > 1) { updateSuggestions(); }
      }
    }
    _tabIndexPrev = _tabIndex;
  }

  void AttributeRenderTask::drawImGuiSuggestor() {
    uint padding = 3;
    uint nSuggestions = _texturesSuggestions.size();
    uint nCols = 6;
    uint nRows = nSuggestions / nCols + 1;
    uint maxRows = 7;
    uint childWidth = 400;
    uint imageWidth = (childWidth - 20) / nCols - padding * 3;
    uint imageHeight = imageWidth * (_params->imgHeight / _params->imgWidth);
    uint childHeight = (imageHeight + padding * 3) * std::min(nRows, maxRows) + 10;

    ImVec2 defaultSpacing = ImGui::GetStyle().ItemSpacing;
    ImGui::GetStyle().ItemSpacing = ImVec2(padding, padding);

    ImGui::BeginChild("Suggestions", ImVec2(childWidth, childHeight), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (uint i = 0; i < nRows; i++) {
      for(uint j = 0; j < nCols; j++) {
        if(i * nCols + j >= nSuggestions || nSuggestions == 0) { break; }
        uint datapointIndex = _indicesSuggestions[i * nCols + j];
        int archetypeIndex = _datapointArchetypeMapping.find(datapointIndex) == _datapointArchetypeMapping.end() ? -1 : (int) _datapointArchetypeMapping[datapointIndex];
        if(archetypeIndex >= 0) {
          ImGui::PushStyleColor(ImGuiCol_Button, _buttonsColors[_archetypeLabels[archetypeIndex]]);
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _buttonsColors[_archetypeLabels[archetypeIndex]]);
        } else {
          ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
        }
        ImGui::ImageButton((void*)(intptr_t)_texturesSuggestions[i * nCols + j], ImVec2(imageWidth, imageHeight), ImVec2(0,0), ImVec2(1,1), padding);
        ImGui::PopStyleColor(2);
        if(ImGui::IsItemHovered()) {
          if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            int archetypeClass = _input.num > 0 ? _input.num : _archetypeClassSelected;
            if(archetypeIndex < 0) {
              addArchetype(datapointIndex, archetypeClass);
            } else {
              bool overwrite = _archetypeLabels[_datapointArchetypeMapping[datapointIndex]] != archetypeClass;
              removeArchetype(datapointIndex);
              if(overwrite) {
                addArchetype(datapointIndex, archetypeClass);
              } else {
                archetypeIndex = -1;
              }
            }
          } else
          if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            _selectedDatapoint = datapointIndex;
          }
          ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures(TabType::eOverlay), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
          ImGui::BeginTooltip();
          ImGui::Text("Datapoint %i\nClass %i\n", datapointIndex, getBufferValue<int>(_minimizationBuffers.labels, datapointIndex));
          if(archetypeIndex >= 0) {
            ImGui::Text("Archetype %c", *_buttons[_archetypeLabels[archetypeIndex]]);
          }
          ImGui::EndTooltip();
        }
        if(j < nCols - 1) { ImGui::SameLine(); }
      }
    }
    if(_selectionCounts[0] >= std::pow(2, _suggestionLevel)) {
      if(ImGui::Button("...", ImVec2(imageWidth + padding * 2, imageHeight + padding * 2))) { updateSuggestions(); }
    }
    ImGui::EndChild();

    ImGui::GetStyle().ItemSpacing = defaultSpacing;

    for(int ac = 0; ac < _params->nArchetypeClasses; ++ac) { // Archetype classes, or buttons, since each archetype class has one button
      ImGui::PushStyleColor(ImGuiCol_Button, _buttonsColors[ac]);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _buttonsColors[ac]);
      if(ImGui::Button(_buttons[ac])) {
        _archetypeClassSelected = ac;
      }
      if(ImGui::IsItemHovered()) {
        ImGui::BeginTooltip(); ImGui::Text("%i archetypes", std::count(_archetypeLabels.begin(), _archetypeLabels.end(), ac)); ImGui::EndTooltip();
      }
      ImGui::PopStyleColor(2);
      ImGui::SameLine();
    }
    if(ImGui::SameLine(); ImGui::Button("Erase")) { clearArchetypes(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Remove current archetypes."); ImGui::EndTooltip(); }
  }

  void AttributeRenderTask::drawImGuiTab(uint tabIndex, const char* text) {
    if (ImGui::BeginTabItem(text)) {
      _tabIndex = tabIndex;
      if(_params->imageDataset) {
        drawImGuiTexture();
      } else {
        drawImPlotBarPlot();
      }
      ImGui::EndTabItem();
    }
  }

  // Gets a specified location in a specified buffer
  template <typename T>
  T AttributeRenderTask::getBufferValue(GLuint buffer, int index) {
    T value;
    glGetNamedBufferSubData(buffer, index * sizeof(T), sizeof(T), &value);
    return value;
  }

  void AttributeRenderTask::drawImGuiTexture() {
    ImGui::Spacing();
    ImGui::ImageButton((void*)(intptr_t)_textures[_tabIndex], ImVec2(300, 300), ImVec2(0,0), ImVec2(1,1), 0);

    if(ImGui::IsItemHovered()) {
      ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures(TabType::eOverlay), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
      uint teXel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 300 * _params->imgWidth;
      uint teYel = (ImGui::GetMousePos().y - ImGui::GetItemRectMin().y) / 300 * _params->imgHeight;
      uint hoveredTexel = teYel * _params->imgWidth + teXel;

      ImGui::BeginTooltip();
      ImGui::Text("Attribute: #%d", hoveredTexel);
      ImGui::Text("Weight: %0.2f", getBufferValue<float>(_similaritiesBuffers.attributeWeights, hoveredTexel));
      float texelValue = getBufferValue<float>(_buffersTextureData[_tabIndex], hoveredTexel * _params->imgDepth);
      ImGui::Text(_promptsValuetype[_tabIndex].c_str(), texelValue);
      std::string pairtype = "";
      if(_tabIndex >= TabType::ePairwiseDiffsAll && _tabIndex <= TabType::ePairwiseDiffsIntraselection) {
        pairtype += _vizAllPairs ? "pairs" : "neighbours";
      }
      ImGui::Text(("Averaged over: " + _promptsDenomtype[_tabIndex] + pairtype).c_str(), _denominators[_tabIndex]);
      ImGui::EndTooltip();

      if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) { _draggedTexel = hoveredTexel; }
      else { _draggedTexel = -1; }

      if(_input.z) {
        std::cout << "\n" << sumWeightedAttributeValues(_tabIndex) << "\n";
      }
    } else {
      _draggedTexel = -1;
    }

    ImGui::SameLine(); ImGui::VSliderFloat("##v", ImVec2(40, 300), &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attr\nWght\n%.2f");
    ImGui::SameLine(); ImGui::VSliderInt("##i", ImVec2(40, 300), &_brushRadius, 0, 10, "Brsh\nSize\n%i");

    if(ImGui::Button("Autoweigh")) { autoweighAttributes(_tabIndex, _autoselectPercentage); }
    ImGui::SameLine(); ImGui::Text("top");
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.25f);
    ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    ImGui::SameLine(); ImGui::Separator();

    if(_tabIndex >= TabType::ePairwiseDiffsAll && _tabIndex <= TabType::ePairwiseDiffsIntraselection) {
      bool vizAllPairsPrev = _vizAllPairs;
      ImGui::SameLine(); ImGui::Checkbox("All pairs", &_vizAllPairs);
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Crash sensitive: Windows may terminate the program if this takes too long."); ImGui::EndTooltip(); }
      if(_vizAllPairs != vizAllPairsPrev) { updateVisualizations(_selectionCounts); }
    }
    
    if(                   ImGui::Button("Clear weights")) { clearAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { invertAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { refineAttributeWeights(_tabIndex); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Refines current attribute weights."); ImGui::EndTooltip(); }

    ImGui::Text("Recalc simil.");
    if(ImGui::SameLine(); ImGui::Button("Ratio")) { _buttonPressed = 2; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Range")) { _buttonPressed = 25; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Resemble")) { _buttonPressed = 26; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by comparing."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Reset")) { _buttonPressed = 3; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Reinstates the original similarities calculated from the dataset."); ImGui::EndTooltip(); }
    if(!_separationMode) {
      if(ImGui::SameLine(); ImGui::Button("Start")) { _buttonPressed = 1000; _separationMode = true; }
    } else {
      if(ImGui::SameLine(); ImGui::Button("Stop")) { _buttonPressed = 1001; _separationMode = false; }
    }

    // Draw dragselected attributes in texture
    if(_draggedTexel >= 0 && _draggedTexel != _draggedTexelPrev) {
      if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        brushTexels(_draggedTexel, _brushRadius, _attributeWeight);
      } else
      if(ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        eraseTexels(_draggedTexel, _brushRadius);
      }
      _draggedTexelPrev = _draggedTexel;
    }
  }

  void AttributeRenderTask::drawImPlotBarPlot() {
    ImPlot::SetNextAxesLimits(0.f, (float) _params->nHighDims, 0.f, 1.f);
    if (ImPlot::BeginPlot("##", ImVec2(400, 200), ImPlotFlags_NoFrame | ImPlotFlags_Crosshairs | ImPlotFlags_CanvasOnly)) {
      std::vector<float> xs(_params->nHighDims);
      std::iota(xs.begin(), xs.end(), 0); // Fills xs with 0..nHighDims-1
      std::vector<float> ys(_params->nHighDims);
      glGetNamedBufferSubData(_buffersTextureData[_tabIndex], 0, _params->nHighDims * sizeof(float), ys.data());

      ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_AutoFit); // ImPlot 0.14 or later
      ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_AutoFit | (_input.x ? ImPlotAxisFlags_Lock : 0) | ImPlotAxisFlags_NoDecorations); // ImPlot 0.14 or later

      ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(1), 1.f);
      ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(1), 0.f);
      ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);

      // Plotting the weights
      if(ImPlot::IsPlotHovered()) {
        glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, 0, _params->nHighDims * sizeof(float), ys.data());
        float max = ImPlot::GetPlotLimits().Max().y;
        float min = ImPlot::GetPlotLimits().Min().y;
        std::vector<float> ys_min(_params->nHighDims);
        std::vector<float> ys_max(_params->nHighDims);
        for(uint i = 0; i < _params->nHighDims; ++i) {
          ys_min[i] = ys[i] * min;
          ys_max[i] = ys[i] * max;
        }

        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
        ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(0), 0.f);
        ImPlot::PlotBars("Weights", xs.data(), ys_max.data(), _params->nHighDims, 1.f);
        if(max * min < 0) { // If max and min have a different sign, also plot the other side of the axis
          ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
          ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(0), 0.f);
          ImPlot::PlotBars("Weights", xs.data(), ys_min.data(), _params->nHighDims, 1.f);
        }

        uint hoveredTexel = std::max(std::round((float) ImPlot::GetPlotMousePos().x), 0.f);

        ImGui::BeginTooltip();
        ImGui::Text("Attribute: #%d", hoveredTexel);
        ImGui::Text("Weight: %0.2f", getBufferValue<float>(_similaritiesBuffers.attributeWeights, hoveredTexel));
        float texelValue = getBufferValue<float>(_buffersTextureData[_tabIndex], hoveredTexel * _params->imgDepth);
        ImGui::Text(_promptsValuetype[_tabIndex].c_str(), texelValue);
        std::string pairtype = "";
        if(_tabIndex >= TabType::ePairwiseDiffsAll && _tabIndex <= TabType::ePairwiseDiffsIntraselection) {
          pairtype += _vizAllPairs ? "pairs" : "neighbours";
        }
        ImGui::Text(("Averaged over: " + _promptsDenomtype[_tabIndex] + pairtype).c_str(), _denominators[_tabIndex]);
        ImGui::EndTooltip();

        if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) { _draggedTexel = hoveredTexel; }
        else { _draggedTexel = -1; }
      } else {
        _draggedTexel = -1;
      }
      ImPlot::EndPlot();
    }

    ImGui::SliderFloat("##v", &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attribute weight %.2f");
    ImGui::SliderInt("##i", &_brushRadius, 0, 10, "Brush size %i");

    if(ImGui::Button("Autoweigh")) { autoweighAttributes(_tabIndex, _autoselectPercentage); }
    ImGui::SameLine(); ImGui::Text("top");
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.25f);
    ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    ImGui::SameLine(); ImGui::Separator();

    if(_tabIndex >= TabType::ePairwiseDiffsAll && _tabIndex <= TabType::ePairwiseDiffsIntraselection) {
      bool vizAllPairsPrev = _vizAllPairs;
      ImGui::SameLine(); ImGui::Checkbox("All pairs", &_vizAllPairs);
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Crash sensitive: Windows may terminate the program if this takes too long."); ImGui::EndTooltip(); }
      if(_vizAllPairs != vizAllPairsPrev) { updateVisualizations(_selectionCounts); }
    }
    
    if(                   ImGui::Button("Clear weights")) { clearAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { invertAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { refineAttributeWeights(_tabIndex); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Refines current attribute weights."); ImGui::EndTooltip(); }

    ImGui::Text("Recalc simil.");
    if(ImGui::SameLine(); ImGui::Button("Ratio")) { _buttonPressed = 2; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Range")) { _buttonPressed = 25; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Resemble")) { _buttonPressed = 26; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by comparing."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Reset")) { _buttonPressed = 3; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Reinstates the original similarities calculated from the dataset."); ImGui::EndTooltip(); }

    // Draw dragselected attributes in barplot
    if(_draggedTexel >= 0 && _draggedTexel != _draggedTexelPrev) {
      if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        for(int i = -_brushRadius; i <= _brushRadius; ++i) {
          if(_draggedTexel + i < 0 || _draggedTexel + i >= _params->nHighDims) { continue; }
          setTexelWeight(_draggedTexel + i, _attributeWeight);
        }
      }
      _draggedTexelPrev = _draggedTexel;
    }
  }

  // Draws the class list
  void AttributeRenderTask::drawImGuiComponentSecondary() {
    _classButtonPressed = -1;
    _setChanged = false;
    if (ImGui::CollapsingHeader("Classes", ImGuiTreeNodeFlags_DefaultOpen)) {
      for(uint i = 0; i < _params->nClasses; ++i) {
        if(ImGui::ImageButton((void*)(intptr_t)_classTextures[i], ImVec2(19, 19), ImVec2(0,0), ImVec2(1,1), 0)) { _classButtonPressed = i; }
        ImVec4 color = ImVec4(_classColors[i].x / 400.f, _classColors[i].y / 400.f, _classColors[i].z / 400.f, 1.f);
        std::string leadingZeros = i < 10 ? "0" : "";
        std::string text = leadingZeros + std::to_string(i) + " | " + std::to_string(_classCounts[i]) + " / " + std::to_string(_classCountsSelected[i]) + " " + _classNames[i];
        if(ImGui::SameLine(); ImGui::ColorEdit3(text.c_str(), (float*) &color, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder)) {
          glm::vec3 colorUpdated(color.x * 400.f, color.y * 400.f, color.z * 400.f);
          _classColors[i] = colorUpdated;
          glNamedBufferSubData(_bufferClassColors, i * sizeof(glm::vec3), sizeof(glm::vec3), &colorUpdated);
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 100);
        if(!_classIsSet[i]) {
          std::string label = "Track##" + std::to_string(i);
          if(ImGui::Button(label.c_str(), ImVec2(70, 0))) { setClass(i); }
        } else {
          std::string label = "Untrack##" + std::to_string(i);
          if(ImGui::Button(label.c_str(), ImVec2(70, 0))) { unsetClass(i); }
        }
      }
    }
  }

  void AttributeRenderTask::setClass(int c) {
    if(_classesSet.size() == 2) {
      int classPopped = *_classesSet.begin();
      _classesSet.erase(_classesSet.begin());
      _classIsSet[classPopped] = false;
    }
    _classIsSet[c] = true;
    _classesSet.insert(c);
    _setChanged = true;
  }

  void AttributeRenderTask::unsetClass(int c) {
    _classIsSet[c] = false;
    _classesSet.erase(c);
    _setChanged = true;
  }

  void AttributeRenderTask::updateVisualizations(std::vector<uint> selectionCounts) {
    _selectionCounts = selectionCounts;
    for(uint i = 0; i < 2; ++i) { _denominators[i * 2] = _selectionCounts[i]; _denominators[i * 2 + 1] = _selectionCounts[i]; }

    // Calculate selection average and/or variance per attribute
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_similaritiesBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, _selectionCounts[i], _buffersTextureData[i * 2], i + 1);
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_similaritiesBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, _selectionCounts[i], _buffersTextureData[i * 2 + 1], i + 1, -1, _buffersTextureData[i * 2], true); // Variance
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().operate(0, _buffersTextureData[i], _buffersTextureData[i+2], _params->nHighDims, _buffersTextureData[i+4]);
    }

    // Compute pairwise attribute differences if one of the relevant texture tabs is open
    if(_tabIndex >= TabType::ePairwiseDiffsAll && _tabIndex <= TabType::ePairwiseDiffsIntraselection) {
      glClearNamedBufferData(_buffers(BufferType::ePairwiseAttrDists), GL_R32F, GL_RED, GL_FLOAT, NULL);
      glClearNamedBufferData(_buffers(BufferType::ePairsSelectedPerDatapoint), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);

      auto &program = _programs(ProgramType::ePairwiseAttrDiffsComp);
      program.bind();

      uint selectionCount = _selectionCounts[0] + _selectionCounts[1];
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      program.template uniform<bool>("vizAllPairs", _vizAllPairs);
      if(_tabIndex == TabType::ePairwiseDiffsInterclass || _tabIndex == TabType::ePairwiseDiffsIntraclass) {
        std::pair<int, int>classes(*_classesSet.begin(), *std::next(_classesSet.begin()));
        program.template uniform<bool>("discriminate", true);
        program.template uniform<int>("maskValueA", classes.first);
        program.template uniform<int>("maskValueB", classes.second);
        program.template uniform<bool>("inter", _tabIndex == TabType::ePairwiseDiffsInterclass);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.labels);
      } else
      if(_tabIndex == TabType::ePairwiseDiffsInterselection || _tabIndex == TabType::ePairwiseDiffsIntraselection) {
        program.template uniform<bool>("discriminate", true);
        program.template uniform<int>("maskValueA", 1);
        program.template uniform<int>("maskValueB", 2);
        program.template uniform<bool>("inter", _tabIndex == TabType::ePairwiseDiffsInterselection);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
      } else {
        program.template uniform<bool>("discriminate", false);
      }

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.selection);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similaritiesBuffers.dataset);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairwiseAttrDists));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsSelectedPerDatapoint));

      // Dispatch shader in batches of batchSize attributes
      glDispatchCompute(ceilDiv(_params->n * _params->nHighDims, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      uint nSelectedPairs = dh::util::BufferTools::instance().reduce<int>(_buffers(BufferType::ePairsSelectedPerDatapoint), 0, _params->n, _minimizationBuffers.selection);
      dh::util::BufferTools::instance().averageTexturedata(_buffers(BufferType::ePairwiseAttrDists), _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, nSelectedPairs, _buffersTextureData[_tabIndex]);
      _denominators[_tabIndex] = nSelectedPairs;
      glAssert();
    }

    for(uint c = 0; c < _params->nClasses; ++c) {
      _classCountsSelected[c] = dh::util::BufferTools::instance().reduce<int>(_minimizationBuffers.labels, 3, _params->n, _minimizationBuffers.selection, c);
    }

    if(_params->imageDataset) { copyTextureDataToTextures(); }
  }

  void AttributeRenderTask::clearSelection() {
    for(uint i = 0; i < _buffersTextureData.size() - 3; ++i) {
      glClearNamedBufferData(_buffersTextureData[i], GL_R32F, GL_RED, GL_FLOAT, NULL);
    }
    _classCountsSelected = std::vector<uint>(_params->nClasses, 0);
    _selectionCounts = std::vector<uint>(2, 0);

    if(_params->imageDataset) { copyTextureDataToTextures(); }
  }

  void AttributeRenderTask::clearSuggestions() {
    _suggestionLevel = 1;
    glDeleteTextures(_texturesSuggestions.size(), _texturesSuggestions.data());
    _texturesSuggestions.clear();
    _indicesSuggestions.clear();
  }

  // void AttributeRenderTask::assess(uint symmetricSize) {
  //   // Create and initialize temp buffers
  //   GLuint watBuffer;
  //   glCreateBuffers(1, &watBuffer);
  //   std::vector<uint> attributeIndices;
  //   if(_weightedAttributeIndices.size() > 0) {
  //     attributeIndices = std::vector<uint>(_weightedAttributeIndices.begin(), _weightedAttributeIndices.end());
  //   } else {
  //     attributeIndices = std::vector<uint>(_params->nHighDims);
  //     std::iota(attributeIndices.begin(), attributeIndices.end(), 0);
  //   }
  //   glNamedBufferStorage(watBuffer, attributeIndices.size() * sizeof(uint), attributeIndices.data(), 0);

  //   GLuint classGuesses;
  //   glCreateBuffers(1, &classGuesses);
  //   std::vector<uint> zeros(_params->n, 0);
  //   glNamedBufferStorage(classGuesses, _params->n * sizeof(uint), zeros.data(), 0);

  //   std::pair<int, int> classes(*_classesSet.begin(), *std::next(_classesSet.begin()));

  //   auto &program = _programs(ProgramType::dGuessClasses);
  //   program.bind();
  //   glAssert();

  //   program.template uniform<uint>("nPoints", _params->n);
  //   program.template uniform<uint>("nHighDims", _params->nHighDims);
  //   program.template uniform<uint>("nWeightedAttribs", attributeIndices.size());
  //   program.template uniform<uint>("classA", classes.first);
  //   program.template uniform<uint>("classB", classes.second);

  //   // Set buffer bindings
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTextureData(TabType::eSnapslotA));
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTextureData(TabType::eSnapslotB));
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, watBuffer);
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.dataset);
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.attributeWeights);
  //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, classGuesses);

  //   // Dispatch shader
  //   glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
  //   glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  //   glAssert();

  //   std::vector<uint> selc(_params->n);
  //   glGetNamedBufferSubData(_minimizationBuffers.selection, 0, _params->n * sizeof(uint), selc.data());
  //   std::vector<int> labl(_params->n);
  //   glGetNamedBufferSubData(_minimizationBuffers.labels, 0, _params->n * sizeof(int), labl.data());
  //   std::vector<uint> gues(_params->n);
  //   glGetNamedBufferSubData(classGuesses, 0, _params->n * sizeof(uint), gues.data());

  //   uint countClassCorrect = 0;
  //   int classCorrect = 1;
  //   int classIncorrect = 0;
  //   for(uint i = 0; i < _params->n; ++i) {
  //     if(selc[i] != 1 || (labl[i] != classes.first && labl[i] != classes.second)) { continue; }

  //     if(labl[i] == gues[i]) {
  //       countClassCorrect++;
  //       glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &classCorrect);
  //     } else {
  //       glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &classIncorrect);
  //     }

  //     glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &gues[i]);
  //     glAssert();
  //   }

  //   std::cout << "\nDatapoint class guesses: " << countClassCorrect << " / " << _selectionCounts[0] << " = " << (float) countClassCorrect / (float) _selectionCounts[0] << "\n";

  //   std::vector<uint> neig(symmetricSize);
  //   glGetNamedBufferSubData(_similaritiesBuffers.neighbors, 0, symmetricSize * sizeof(uint), neig.data());
  //   std::vector<uint> layo(_params->n * 2);
  //   glGetNamedBufferSubData(_similaritiesBuffers.layout, 0, _params->n * 2 * sizeof(uint), layo.data());

  //   uint countNeighbortypeCorrect = 0;
  //   uint countNeighbortypeCorrectInter = 0;
  //   uint countNeighbortypeCorrectIntra = 0;
  //   uint countNeighbors = 0;
  //   uint countNeighborsInter = 0;
  //   uint countNeighborsIntra = 0;
  //   for(uint i = 0; i < _params->n; ++i) {
  //     if(selc[i] != 1 || (labl[i] != classes.first && labl[i] != classes.second)) { continue; }
  //     for(uint ij = layo[i*2+0]; ij < layo[i*2+0] + layo[i*2+1]; ++ij) {
  //       uint j = neig[ij];
  //       if(selc[j] != 1 || (labl[j] != classes.first && labl[j] != classes.second)) { continue; }
  //       countNeighbors++;
  //       if(labl[i] == labl[j]) { countNeighborsIntra++; } else
  //       if(labl[i] != labl[j]) { countNeighborsInter++; }

  //       if((gues[i] == gues[j]) == (labl[i] == labl[j])) {
  //         countNeighbortypeCorrect++;
  //         if(labl[i] == labl[j]) { countNeighbortypeCorrectIntra++; } else
  //         if(labl[i] != labl[j]) { countNeighbortypeCorrectInter++; }
  //       }
  //     }
  //   }

  //   std::cout << "Neighbour type guesses: " << countNeighbortypeCorrect << " / " << countNeighbors << " = " << (float) countNeighbortypeCorrect / (float) countNeighbors << "\n";
  //   std::cout << "Neighbour type guesses (inter): " << countNeighbortypeCorrectInter << " / " << countNeighborsInter << " = " << (float) countNeighbortypeCorrectInter / (float) countNeighborsInter << "\n";
  //   std::cout << "Neighbour type guesses (intra): " << countNeighbortypeCorrectIntra << " / " << countNeighborsIntra << " = " << (float) countNeighbortypeCorrectIntra / (float) countNeighborsIntra << "\n";
  // }

} // dh::vis