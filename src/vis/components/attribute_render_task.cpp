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
#include <numeric>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
#include <implot.h>
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/vis/components/attribute_render_task.hpp"
#include "dh/util/io.hpp" //

namespace dh::vis {

  std::array<const std::string, 9> promptsValuetype = {"Mean: %0.2f",
                                                 "Variance: %0.2f",
                                                 "Mean: %0.2f",
                                                 "Variance: %0.2f",
                                                 "Difference in mean: %0.2f",
                                                 "Difference in variance: %0.2f",
                                                 "Pairwise difference (all): %0.2f",
                                                 "Pairwise difference (interclass): %0.2f",
                                                 "Pairwise difference (intraclass): %0.2f"};

  std::array<const std::string, 9> promptsDenomtype = {"%u primary selected datapoints",
                                                       "%u primary selected datapoints",
                                                       "%u secondary selected datapoints",
                                                       "%u secondary selected datapoints",
                                                       "",
                                                       "",
                                                       "%u selected ",
                                                       "%u selected interclass ",
                                                       "%u selected intraclass "};
  
  AttributeRenderTask::AttributeRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  AttributeRenderTask::AttributeRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, sne::SimilaritiesBuffers similaritiesBuffers, GLuint colorBuffer, const int* labelPtr)
  : RenderTask(priority, "AttributeRenderTask"),
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _similaritiesBuffers(similaritiesBuffers),
    _colorBuffer(colorBuffer),
    _params(params),
    _selectionCounts(2, 0),
    _draggedTexel(-1),
    _buttonPressed(0),
    _attributeWeight(0.f),
    _brushRadius(1),
    _similarityWeight(2.f),
    _autoselectPercentage(0.025f),
    _currentTabUpper(0),
    _currentTabLower(0),
    _classIsSet(_params->nClasses, false),
    _classesSet(),
    _setChanged(false),
    _classButtonPressed(-1),
    _vizAllPairs(false),
    _denominators(9, 0) {

    // Initialize shader program
    {
      _programs(ProgramType::ePairwiseAttrDiffsNeiComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/pairwise_attr_diffs_nei.comp"));
      _programs(ProgramType::ePairwiseAttrDiffsAllComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/pairwise_attr_diffs_all.comp"));
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
      glCreateBuffers(_buffersTextureData.size(), _buffersTextureData.data());
      for(uint i = 0; i < _buffersTextureData.size() - 1; ++i) {
        glNamedBufferStorage(_buffersTextureData[i], _params->nTexels * _params->imgDepth * sizeof(float), nullptr, 0);
      }
      glNamedBufferStorage(_buffersTextureData(TextureType::eOverlay), _params->nTexels * 4 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
      mirrorWeightsToOverlay();
    }

    // Count classes
    _classCountsSelected = std::vector<uint>(_params->nClasses, 0);
    _classCounts = std::vector<uint>(_params->nClasses);
    for(uint i = 0; i < _params->nClasses; ++i) {
      // uint classCount = std::count(labelPtr, labelPtr + _params->n, i);
      uint classCount = dh::util::BufferTools::instance().reduce<uint>(_minimizationBuffers.labels, 3, _params->n, 0, i);
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
        
        dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.labels, i, _classCounts[i], classTextureBuffers[i]);
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
      }
      glTextureParameteri(_textures(TextureType::eOverlay), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eOverlay), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(_textures(TextureType::eOverlay), 1, GL_RGBA8, _params->imgWidth, _params->imgHeight);
    }

    _classNames = std::vector<std::string>(_params->nClasses, "");
    dh::util::readTxtClassNames(_params->datasetName + ".txt", _classNames, _params->nClasses);

    _colors = std::vector<glm::vec4>(_params->nClasses);
    glGetNamedBufferSubData(_colorBuffer, 0, _params->nClasses * sizeof(glm::vec4), _colors.data());

    _isInit = true;
  }

  AttributeRenderTask::~AttributeRenderTask() {
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteBuffers(_classTextures.size(), _classTextures.data());
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
    glNamedBufferSubData(_buffersTextureData(TextureType::eOverlay), texelIndex * 4 * sizeof(float), 4 * sizeof(float), color.data());
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
  }

  void AttributeRenderTask::clearAttributeWeights() {
    const std::vector<float> ones(_params->nHighDims, 1.0f);
    glClearNamedBufferData(_similaritiesBuffers.attributeWeights, GL_R32F, GL_RED, GL_FLOAT, ones.data());
    mirrorWeightsToOverlay();
    _weightedAttributeIndices = std::set<uint>();
  }

  void AttributeRenderTask::invertAttributeWeights() {
    for(uint i = 0; i < _params->nTexels; ++i) {
      float weightTexel = getTexelWeight(i);
      float weightCurrent = _attributeWeight;
      float weight = 1.f + weightCurrent - weightTexel;
      weight = std::clamp(weight, 0.f, _params->maxAttributeWeight);
      setTexelWeight(i, weight);
    }
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

  void AttributeRenderTask::render(glm::mat4 model_view, glm::mat4 proj) {
    if (!enable) {
      return;
    }

    // Copy texture data to textures
    if(_params->imageDataset) {
      for(uint i = 0; i < _textures.size() - 1; ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTextureData[i]);
        GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
        glTextureSubImage2D(_textures[i], 0, 0, 0, _params->imgWidth, _params->imgHeight, format, GL_FLOAT, 0);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffersTextureData[_buffersTextureData.size()-1]);
      glTextureSubImage2D(_textures(TextureType::eOverlay), 0, 0, 0, _params->imgWidth, _params->imgHeight, GL_RGBA, GL_FLOAT, 0);
    }
    glAssert();

    if(_params->imageDataset) {
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
    } else {
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
    glAssert();
  }

  void AttributeRenderTask::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Attribute render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::BeginTabBar("Selection tabs")) {
        if (ImGui::BeginTabItem(_selectionCounts[1] > 0 ? "Selection pri" : "Selection")) {
          _currentTabUpper = 0;
          ImGui::EndTabItem();
        }

        if(_selectionCounts[1] > 0) {
          if (ImGui::BeginTabItem("Selection sec")) {
            _currentTabUpper = 1;
            ImGui::EndTabItem();
          }

          if (ImGui::BeginTabItem("Selection diff")) {
            _currentTabUpper = 2;
            ImGui::EndTabItem();
          }
        }

        if (ImGui::BeginTabItem("Pairwise")) {
          _currentTabUpper = 3;
          ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
      }

      _draggedTexel = -1;

      if (ImGui::BeginTabBar("Selection attributes type tabs")) {
        if(_currentTabUpper <= 2) {
          drawImGuiTab(_currentTabUpper, 0, "Average");
          drawImGuiTab(_currentTabUpper, 1, "Variance");
        } else
        if(_currentTabUpper == 3) {
          drawImGuiTab(_currentTabUpper, 0, "All");
          if(_classesSet.size() == 2) {
            drawImGuiTab(_currentTabUpper, 1, "Interclass");
            drawImGuiTab(_currentTabUpper, 2, "Intraclass");
          }
        }
        ImGui::EndTabBar();
      }
    }

    // Force selection re-evaluation when switching to the pairwise attr diff tab
    if(_currentTabUpper == 3 && currentTabIndex() != _previousTabIndex || _setChanged) { update(_selectionCounts); }
    _previousTabIndex = currentTabIndex();
  }

  void AttributeRenderTask::drawImGuiTab(uint tabUpper, uint tabLower, const char* text) {
    if (ImGui::BeginTabItem(text)) {
      _currentTabLower = tabLower;
      if(_params->imageDataset) {
        drawImGuiTexture(currentTabIndex());
      } else {
        drawImPlotBarPlot(currentTabIndex());
      }
      ImGui::EndTabItem();
    }
  }

  // Gets a specified location in a specified buffer
  float AttributeRenderTask::getBufferValue(GLuint buffer, int index) {
    float value;
    glGetNamedBufferSubData(buffer, index * sizeof(float), sizeof(float), &value);
    return value;
  }

  void AttributeRenderTask::drawImGuiTexture(uint index) {
    ImGui::Spacing();
    ImGui::ImageButton((void*)(intptr_t)_textures[index], ImVec2(300, 300), ImVec2(0,0), ImVec2(1,1), 0);

    if(ImGui::IsItemHovered()) {
      ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures(TextureType::eOverlay), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
      uint teXel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 300 * _params->imgWidth;
      uint teYel = (ImGui::GetMousePos().y - ImGui::GetItemRectMin().y) / 300 * _params->imgHeight;
      uint hoveredTexel = teYel * _params->imgWidth + teXel;

      ImGui::BeginTooltip();
      ImGui::Text("Attribute: #%d", hoveredTexel);
      ImGui::Text("Weight: %0.2f", getBufferValue(_similaritiesBuffers.attributeWeights, hoveredTexel));
      float texelValue = getBufferValue(_buffersTextureData[index], hoveredTexel * _params->imgDepth);
      ImGui::Text(promptsValuetype[index].c_str(), texelValue);
      std::string pairtype = "";
      if(index > 5) { pairtype += _vizAllPairs ? "pairs" : "neighbours"; }
      ImGui::Text(("Averaged over: " + promptsDenomtype[index] + pairtype).c_str(), _denominators[index]);
      ImGui::EndTooltip();

      if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
      else { _draggedTexel = -1; }

      if(_input.z) {
        std::cout << "\n" << sumWeightedAttributeValues(index) << "\n";
      }
    } else {
      _draggedTexel = -1;
    }

    ImGui::SameLine(); ImGui::VSliderFloat("##v", ImVec2(40, 300), &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attr\nWght\n%.2f");
    ImGui::SameLine(); ImGui::VSliderInt("##i", ImVec2(40, 300), &_brushRadius, 0, 10, "Brsh\nSize\n%i");

    if(ImGui::Button("Autoweigh")) { autoweighAttributes(currentTabIndex(), _autoselectPercentage); }
    ImGui::SameLine(); ImGui::Text("top");
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.25f);
    ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    ImGui::SameLine(); ImGui::Separator();
    bool vizAllPairsPrev = _vizAllPairs;
    ImGui::SameLine(); ImGui::Checkbox("All pairs", &_vizAllPairs);
    if(_vizAllPairs != vizAllPairsPrev) { update(_selectionCounts); }
    
    if(                   ImGui::Button("Clear weights")) { clearAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { invertAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { refineAttributeWeights(currentTabIndex()); }
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

    ImGui::Dummy(ImVec2(193.0f, 13.0f)); ImGui::SameLine();
    for(int i = -3; i < -1; ++i) {
      ImGui::ImageButton((void*)(intptr_t)_textures[_buffersTextureData.size()+i], ImVec2(28, 28), ImVec2(0,0), ImVec2(1,1), 0); ImGui::SameLine();
      if(ImGui::IsItemHovered()) {
        ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures(TextureType::eOverlay), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
        if(ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          glCopyNamedBufferSubData(_buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i], 0, 0, _params->nHighDims * sizeof(float));
        } else
        if(ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
          if(_classesSet.size() == 1) {
            dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, _selectionCounts[0], _buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i], false, *_classesSet.begin(), _minimizationBuffers.labels);
          } else {
            dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, _selectionCounts[0], _buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i], false);
          }
        }
      }
    }
  }

  void AttributeRenderTask::drawImPlotBarPlot(uint index) {
    ImPlot::SetNextAxesLimits(0.f, (float) _params->nHighDims, 0.f, 1.f);
    if (ImPlot::BeginPlot("##", ImVec2(400, 200), ImPlotFlags_NoFrame | ImPlotFlags_Crosshairs | ImPlotFlags_CanvasOnly)) {
      std::vector<float> xs(_params->nHighDims);
      std::iota(xs.begin(), xs.end(), 0); // Fills xs with 0..nHighDims-1
      std::vector<float> ys(_params->nHighDims);
      glGetNamedBufferSubData(_buffersTextureData[index], 0, _params->nHighDims * sizeof(float), ys.data());

      ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | (_input.x ? ImPlotAxisFlags_AutoFit : 0)); // ImPlot 0.14 or later
      ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_NoHighlight | ImPlotAxisFlags_NoSideSwitch | (_input.x ? ImPlotAxisFlags_Lock : 0) | ImPlotAxisFlags_NoDecorations); // ImPlot 0.14 or later

      ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(1), 1.f);
      ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(1), 0.f);
      ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);

      if(ImPlot::IsPlotHovered() && _input.x) {
        glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, 0, _params->nHighDims * sizeof(float), ys.data());
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
        ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(0), 0.f);
        ImPlot::PlotBars("Weights", xs.data(), ys.data(), _params->nHighDims, 1.f);

        uint hoveredTexel = ImPlot::GetPlotMousePos().x;

        ImGui::BeginTooltip();
        ImGui::Text("Attribute: #%d", hoveredTexel);
        ImGui::Text("Weight: %0.2f", getBufferValue(_similaritiesBuffers.attributeWeights, hoveredTexel));
        float texelValue = getBufferValue(_buffersTextureData[index], hoveredTexel * _params->imgDepth);
        ImGui::Text(promptsValuetype[index].c_str(), texelValue);
        std::string pairtype = "";
        if(index > 5) { pairtype += _vizAllPairs ? "pairs" : "neighbours"; }
        ImGui::Text(("Averaged over: " + promptsDenomtype[index] + pairtype).c_str(), _denominators[index]);
        ImGui::EndTooltip();

        if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
        else { _draggedTexel = -1; }
      } else {
        _draggedTexel = -1;
      }
      ImPlot::EndPlot();
    }

    ImGui::SliderFloat("##v", &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attribute weight %.2f");
    ImGui::SliderInt("##i", &_brushRadius, 0, 10, "Brush size %i");

    if(ImGui::Button("Autoweigh")) { autoweighAttributes(currentTabIndex(), _autoselectPercentage); }
    ImGui::SameLine(); ImGui::Text("top");
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.25f);
    ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    ImGui::SameLine(); ImGui::Separator();
    bool vizAllPairsPrev = _vizAllPairs;
    ImGui::SameLine(); ImGui::Checkbox("All pairs", &_vizAllPairs);
    if(_vizAllPairs != vizAllPairsPrev) { update(_selectionCounts); }
    
    if(                   ImGui::Button("Clear weights")) { clearAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { invertAttributeWeights(); }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { refineAttributeWeights(currentTabIndex()); }
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

    ImGui::Dummy(ImVec2(193.0f, 13.0f)); ImGui::SameLine();
    std::array<const char*, 9> buttons = {" A ", " B "};
    for(int i = 0; i < 2; ++i) {
      ImGui::Button(buttons[i]); ImGui::SameLine();
      if(ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        glCopyNamedBufferSubData(_buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i-3], 0, 0, _params->nHighDims * sizeof(float));
      } else
      if(ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, _selectionCounts[0], _buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i-3]);
      }
    }
  }

  // Draws the class list
  void AttributeRenderTask::drawImGuiComponentSecondary() {
    _classButtonPressed = -1;
    _setChanged = false;
    if (ImGui::CollapsingHeader("Classes", ImGuiTreeNodeFlags_DefaultOpen)) {
      for(uint i = 0; i < _params->nClasses; ++i) {
        if(ImGui::ImageButton((void*)(intptr_t)_classTextures[i], ImVec2(19, 19), ImVec2(0,0), ImVec2(1,1), 0)) { _classButtonPressed = i; }
        ImVec4 color = ImVec4(_colors[i].x / 400.f, _colors[i].y / 400.f, _colors[i].z / 400.f, _colors[i].w / 255.f);
        std::string leadingZeros = i < 10 ? "0" : "";
        std::string text = leadingZeros + std::to_string(i) + " | " + std::to_string(_classCounts[i]) + " / " + std::to_string(_classCountsSelected[i]) + " " + _classNames[i];
        if(ImGui::SameLine(); ImGui::ColorEdit3(text.c_str(), (float*) &color, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder)) {
          glm::vec4 colorUpdated = glm::vec4(color.x * 400.f, color.y * 400.f, color.z * 400.f, 255.f);
          _colors[i] = colorUpdated;
          glNamedBufferSubData(_colorBuffer, i * sizeof(glm::vec4), sizeof(glm::vec4), &colorUpdated);
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

  void AttributeRenderTask::update(std::vector<uint> selectionCounts) {
    _selectionCounts = selectionCounts;
    for(uint i = 0; i < 2; ++i) { _denominators[i * 2] = _selectionCounts[i]; _denominators[i * 2 + 1] = _selectionCounts[i];  }

    // Calculate selection average and/or variance per attribute
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, i + 1, _selectionCounts[i], _buffersTextureData[i * 2]);
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, i + 1, _selectionCounts[i], _buffersTextureData[i * 2 + 1], _buffersTextureData[i * 2], true); // Variance
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().difference(_buffersTextureData[i], _buffersTextureData[i+2], _params->nHighDims, _buffersTextureData[i+4]);
    }

    // Compute pairwise attribute differences if relevant texture tabs are open
    if(_currentTabUpper == 3) {
      std::pair<int, int> classes;
      glClearNamedBufferData(_buffers(BufferType::ePairwiseAttrDists), GL_R32F, GL_RED, GL_FLOAT, nullptr);
      glClearNamedBufferData(_similaritiesBuffers.neighborsSelected, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _vizAllPairs ? _programs(ProgramType::ePairwiseAttrDiffsAllComp) : _programs(ProgramType::ePairwiseAttrDiffsNeiComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      if(_classesSet.size() == 2 && _currentTabLower > 0) {
        classes = std::pair<int, int>(*_classesSet.begin(), *std::next(_classesSet.begin()));
        program.template uniform<bool>("classesSet", true);
        program.template uniform<int>("classA", classes.first);
        program.template uniform<int>("classB", classes.second);
        program.template uniform<bool>("inter", _currentTabLower == 1);
      } else {
        program.template uniform<bool>("classesSet", false);
      }

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.dataset);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimizationBuffers.labels);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairwiseAttrDists));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _similaritiesBuffers.neighborsSelected);

      // Dispatch shader in batches of batchSize attributes
      glDispatchCompute(ceilDiv(_params->n * _params->nHighDims, 256u), 1, 1);

      uint nSelectedPairs;
      if(!_vizAllPairs) {
        nSelectedPairs = dh::util::BufferTools::instance().reduce<uint>(_similaritiesBuffers.neighborsSelected, 0, _params->n, _minimizationBuffers.selection, -1, true, _similaritiesBuffers.layout, _similaritiesBuffers.neighbors);
      } else
      if(_currentTabLower == 0) {
        nSelectedPairs = _selectionCounts[0] * (_selectionCounts[0] - 1);
      } else
      if(_currentTabLower == 1) {
        nSelectedPairs = _classCountsSelected[classes.first] * _classCountsSelected[classes.second] * 2;
      } else
      if(_currentTabLower == 2) {
        nSelectedPairs = _classCountsSelected[classes.first] * (_classCountsSelected[classes.first] - 1) + _classCountsSelected[classes.second] * (_classCountsSelected[classes.second] - 1);
      }
      dh::util::BufferTools::instance().averageTexturedata(_buffers(BufferType::ePairwiseAttrDists), _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, nSelectedPairs, _buffersTextureData[currentTabIndex()]);
      _denominators[currentTabIndex()] = nSelectedPairs;
      glAssert();
    }

    for(uint c = 0; c < _params->nClasses; ++c) {
      _classCountsSelected[c] = dh::util::BufferTools::instance().reduce<uint>(_minimizationBuffers.labels, 3, _params->n, _minimizationBuffers.selection, c);
    }

  }

  void AttributeRenderTask::clear() {
    for(uint i = 0; i < _buffersTextureData.size() - 3; ++i) {
      glClearNamedBufferData(_buffersTextureData[i], GL_R32F, GL_RED, GL_FLOAT, nullptr);
    }
    _classCountsSelected = std::vector<uint>(_params->nClasses, 0);
  }

  void AttributeRenderTask::assess(uint symmetricSize) {
    // Create and initialize temp buffers
    GLuint watBuffer;
    glCreateBuffers(1, &watBuffer);
    std::vector<uint> attributeIndices;
    if(_weightedAttributeIndices.size() > 0) {
      attributeIndices = std::vector<uint>(_weightedAttributeIndices.begin(), _weightedAttributeIndices.end());
    } else {
      attributeIndices = std::vector<uint>(_params->nHighDims);
      std::iota(attributeIndices.begin(), attributeIndices.end(), 0);
    }
    glNamedBufferStorage(watBuffer, attributeIndices.size() * sizeof(uint), attributeIndices.data(), 0);

    GLuint classGuesses;
    glCreateBuffers(1, &classGuesses);
    std::vector<uint> zeros(_params->n, 0);
    glNamedBufferStorage(classGuesses, _params->n * sizeof(uint), zeros.data(), 0);

    std::pair<int, int> classes(*_classesSet.begin(), *std::next(_classesSet.begin()));

    auto &program = _programs(ProgramType::dGuessClasses);
    program.bind();
    glAssert();

    program.template uniform<uint>("nPoints", _params->n);
    program.template uniform<uint>("nHighDims", _params->nHighDims);
    program.template uniform<uint>("nWeightedAttribs", attributeIndices.size());
    program.template uniform<uint>("classA", classes.first);
    program.template uniform<uint>("classB", classes.second);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTextureData(TextureType::eSnapslotA));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTextureData(TextureType::eSnapslotB));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, watBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _minimizationBuffers.dataset);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.attributeWeights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, classGuesses);

    // Dispatch shader
    glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();

    std::vector<uint> selc(_params->n);
    glGetNamedBufferSubData(_minimizationBuffers.selection, 0, _params->n * sizeof(uint), selc.data());
    std::vector<int> labl(_params->n);
    glGetNamedBufferSubData(_minimizationBuffers.labels, 0, _params->n * sizeof(int), labl.data());
    std::vector<uint> gues(_params->n);
    glGetNamedBufferSubData(classGuesses, 0, _params->n * sizeof(uint), gues.data());

    uint countClassCorrect = 0;
    int classCorrect = 1;
    int classIncorrect = 0;
    for(uint i = 0; i < _params->n; ++i) {
      if(selc[i] != 1 || (labl[i] != classes.first && labl[i] != classes.second)) { continue; }

      if(labl[i] == gues[i]) {
        countClassCorrect++;
        glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &classCorrect);
      } else {
        glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &classIncorrect);
      }

      // glNamedBufferSubData(_minimizationBuffers.labels, i * sizeof(int), sizeof(int), &gues[i]);
      glAssert();
    }

    std::cout << "\nDatapoint class guesses: " << countClassCorrect << " / " << _selectionCounts[0] << " = " << (float) countClassCorrect / (float) _selectionCounts[0] << "\n";

    std::vector<uint> neig(symmetricSize);
    glGetNamedBufferSubData(_similaritiesBuffers.neighbors, 0, symmetricSize * sizeof(uint), neig.data());
    std::vector<uint> layo(_params->n * 2);
    glGetNamedBufferSubData(_similaritiesBuffers.layout, 0, _params->n * 2 * sizeof(uint), layo.data());

    uint countNeighbortypeCorrect = 0;
    uint countNeighbortypeCorrectInter = 0;
    uint countNeighbortypeCorrectIntra = 0;
    uint countNeighbors = 0;
    uint countNeighborsInter = 0;
    uint countNeighborsIntra = 0;
    for(uint i = 0; i < _params->n; ++i) {
      if(selc[i] != 1 || (labl[i] != classes.first && labl[i] != classes.second)) { continue; }
      for(uint ij = layo[i*2+0]; ij < layo[i*2+0] + layo[i*2+1]; ++ij) {
        uint j = neig[ij];
        if(selc[j] != 1 || (labl[j] != classes.first && labl[j] != classes.second)) { continue; }
        countNeighbors++;
        if(labl[i] == labl[j]) { countNeighborsIntra++; } else
        if(labl[i] != labl[j]) { countNeighborsInter++; }

        if((gues[i] == gues[j]) == (labl[i] == labl[j])) {
          countNeighbortypeCorrect++;
          if(labl[i] == labl[j]) { countNeighbortypeCorrectIntra++; } else
          if(labl[i] != labl[j]) { countNeighbortypeCorrectInter++; }
        }
      }
    }

    std::cout << "Neighbour type guesses: " << countNeighbortypeCorrect << " / " << countNeighbors << " = " << (float) countNeighbortypeCorrect / (float) countNeighbors << "\n";
    std::cout << "Neighbour type guesses (inter): " << countNeighbortypeCorrectInter << " / " << countNeighborsInter << " = " << (float) countNeighbortypeCorrectInter / (float) countNeighborsInter << "\n";
    std::cout << "Neighbour type guesses (intra): " << countNeighbortypeCorrectIntra << " / " << countNeighborsIntra << " = " << (float) countNeighbortypeCorrectIntra / (float) countNeighborsIntra << "\n";
  }

} // dh::vis