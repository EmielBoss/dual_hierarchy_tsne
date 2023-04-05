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
  // Quad vertex position data
  constexpr std::array<glm::vec2, 4> quadPositions = {
    glm::vec2(-1, -1),  // 0
    glm::vec2(1, -1),   // 1
    glm::vec2(1, 1),    // 2
    glm::vec2(-1, 1)    // 3
  };

  // Quad element index data
  constexpr std::array<uint, 6> quadElements = {
    0, 1, 2,  2, 3, 0
  };
  
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
    _plotError(false),
    _currentTabUpper(0),
    _currentTabLower(0),
    _classIsSet(_params->nClasses, false),
    _classesSet(),
    _setChanged(false),
    _classButtonPressed(-1),
    _nSelectedNeighbors(0) {

    // Initialize shader program
    {
      _programs(ProgramType::ePairwiseAttrDiffsNeiComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/pairwise_attr_diffs_nei.comp"));
      _programs(ProgramType::ePairwiseAttrDiffsAllComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/pairwise_attr_diffs_all.comp"));
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
    _classCounts = std::vector<uint>(_params->nClasses);
    for(uint i = 0; i < _params->nClasses; ++i) {
      uint classCount = std::count(labelPtr, labelPtr + _params->n, i);
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
        if(_currentTabUpper < 3) {
          drawImGuiTab(_currentTabUpper, 0, "Average");
          drawImGuiTab(_currentTabUpper, 1, "Variance");
        } else {
          drawImGuiTab(_currentTabUpper, 0, "Between neighbs");
          drawImGuiTab(_currentTabUpper, 1, "Between all");
          drawImGuiTab(_currentTabUpper, 2, "Difference");
        }
        ImGui::EndTabBar();
      }
    }

    // Force selection re-evaluaation when switching to the pairwise attr diff tab
    if(currentTabIndex() > 5 && currentTabIndex() != _previousTabIndex || _setChanged) { update(_selectionCounts); }
    _previousTabIndex = currentTabIndex();
  }

  void AttributeRenderTask::drawImGuiTab(uint tabUpper, uint tabLower, const char* text) {
    if (ImGui::BeginTabItem(text)) {
      if(_params->imageDataset) {
        drawImGuiTexture(tabUpper * 2 + tabLower);
      } else {
        drawImPlotBarPlot(tabUpper);
      }
      _currentTabUpper = tabUpper;
      _currentTabLower = tabLower;
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
      glAssert();
      ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures(TextureType::eOverlay), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
      glAssert();
      _hoveringTexture = true;
      uint teXel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 300 * _params->imgWidth;
      uint teYel = (ImGui::GetMousePos().y - ImGui::GetItemRectMin().y) / 300 * _params->imgHeight;
      uint hoveredTexel = teYel * _params->imgWidth + teXel;

      ImGui::BeginTooltip();
      ImGui::Text("Attribute: #%d", hoveredTexel);
      glAssert();
      ImGui::Text("Weight: %0.2f", getBufferValue(_similaritiesBuffers.attributeWeights, hoveredTexel));
      glAssert();
      std::array<const char*, 9> prompts = {"Mean: %0.2f", "Variance: %0.2f", "Mean: %0.2f", "Variance: %0.2f", "Difference in mean: %0.2f", "Difference in variance: %0.2f", "Difference: %0.2f", "Difference: %0.2f", "Difference between differences: %0.2f"};
      glAssert();
      float texelValue = getBufferValue(_buffersTextureData[index], hoveredTexel * _params->imgDepth);
      glAssert();
      ImGui::Text(prompts[index], texelValue);
      ImGui::EndTooltip();

      if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
      else { _draggedTexel = -1; }
    } else {
      _hoveringTexture = false;
      _draggedTexel = -1;
    }

    ImGui::SameLine(); ImGui::VSliderFloat("##v", ImVec2(40, 300), &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attr\nWght\n%.2f");
    ImGui::SameLine(); ImGui::VSliderInt("##i", ImVec2(40, 300), &_brushRadius, 0, 10, "Brsh\nSize\n%i");

    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
    if(ImGui::Button("Autoweigh")) { autoweighAttributes(currentTabIndex(), _autoselectPercentage); }
    ImGui::SameLine(); ImGui::Text("top"); ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    
    if(                          ImGui::Button("Clear weights")) { // Clear attribute weights
      const std::vector<float> ones(_params->nHighDims, 1.0f);
      glClearNamedBufferData(_similaritiesBuffers.attributeWeights, GL_R32F, GL_RED, GL_FLOAT, ones.data());
      mirrorWeightsToOverlay();
      _weightedAttributeIndices = std::set<uint>();
    }
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
        if(ImGui::IsAnyMouseDown()) {
          glCopyNamedBufferSubData(_buffersTextureData[index], _buffersTextureData[_buffersTextureData.size()+i], 0, 0, _params->nHighDims * sizeof(float));
        }
      }
    }
  }

  void AttributeRenderTask::drawImPlotBarPlot(uint tabUpper) {
    ImPlot::SetNextAxesLimits(0.f, (float) _params->nHighDims, 0.f, 1.f);
    if (ImPlot::BeginPlot("##", ImVec2(400, 200))) {
      std::vector<float> xs(_params->nHighDims);
      std::iota(xs.begin(), xs.end(), 0); // Fills xs with 0..nHighDims-1
      std::vector<float> ys(_params->nHighDims);
      glGetNamedBufferSubData(_buffersTextureData[tabUpper * 2], 0, _params->nHighDims * sizeof(float), ys.data());

      ImPlot::SetupAxis(ImAxis_X1, NULL, (_input.x ? ImPlotAxisFlags_AutoFit : 0)); // ImPlot 0.14 or later
      ImPlot::SetupAxis(ImAxis_Y1, NULL, (_input.x ? ImPlotAxisFlags_Lock : 0) | ImPlotAxisFlags_NoDecorations); // ImPlot 0.14 or later

      ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(1), 1.f);
      ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(1), 0.f);
      ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);

      if(_plotError) {
        std::vector<float> errs(_params->nHighDims);
        glGetNamedBufferSubData(_buffersTextureData[tabUpper * 2 + 1], 0, _params->nHighDims * sizeof(float), errs.data());
        ImPlot::SetNextErrorBarStyle(ImPlot::GetColormapColor(4), 0.1f, 0.1f);
        ImPlot::PlotErrorBars("Average", xs.data(), ys.data(), errs.data(), _params->nHighDims);
        ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);
      }

      if(ImGui::IsItemHovered() && _input.x) {
        glGetNamedBufferSubData(_similaritiesBuffers.attributeWeights, 0, _params->nHighDims * sizeof(float), ys.data());
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
        ImPlot::SetNextLineStyle(ImPlot::GetColormapColor(0), 0.f);
        ImPlot::PlotBars("Weights", xs.data(), ys.data(), _params->nHighDims, 1.f);

        _hoveringTexture = true;
        uint hoveredTexel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 400 * _params->nHighDims;

        ImGui::BeginTooltip();
        ImGui::Text("Attribute: #%d", hoveredTexel);
        ImGui::Text("Weight: %0.2f", getBufferValue(_similaritiesBuffers.attributeWeights, hoveredTexel));
        std::array<const char*, 9> prompts = {"Mean: %0.2f", "Variance: %0.2f", "Mean: %0.2f", "Variance: %0.2f", "Difference in mean: %0.2f", "Difference in variance: %0.2f", "Difference: %0.2f", "Difference: %0.2f", "Difference between differences: %0.2f"};
        float texelValue = getBufferValue(_buffersTextureData[tabUpper * 2], hoveredTexel * _params->imgDepth);
        ImGui::Text(prompts[tabUpper * 2], texelValue);
        ImGui::EndTooltip();

        if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
        else { _draggedTexel = -1; }
      } else {
        _hoveringTexture = false;
        _draggedTexel = -1;
      }
      ImPlot::EndPlot();
    }

    ImGui::Checkbox("Plot error bars", &_plotError);
    ImGui::SliderFloat("##v", &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attribute weight %.2f");
    ImGui::SliderInt("##i", &_brushRadius, 1, 10, "Brush weight %i");

    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
    if(ImGui::Button("Autoweigh")) { _buttonPressed = 15; }
    ImGui::SameLine(); ImGui::Text("top"); ImGui::SameLine(); ImGui::SliderFloat("of attribs", &_autoselectPercentage, 0.0f, 1.f);
    
    if(                          ImGui::Button("Clear weights")) { _buttonPressed = 4; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { _buttonPressed = 5; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { _buttonPressed = 6; }
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
      if(ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
        glCopyNamedBufferSubData(_buffersTextureData[tabUpper * 2], _buffersTextureData[_buffersTextureData.size()+i-3], 0, 0, _params->nHighDims * sizeof(float));
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
        std::string text = leadingZeros + std::to_string(i) + " | " + std::to_string(_classCounts[i]) + " " + _classNames[i];
        if(ImGui::SameLine(); ImGui::ColorEdit3(text.c_str(), (float*) &color, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder)) {
          glm::vec4 colorUpdated = glm::vec4(color.x * 400.f, color.y * 400.f, color.z * 400.f, 255.f);
          _colors[i] = colorUpdated;
          glNamedBufferSubData(_colorBuffer, i * sizeof(glm::vec4), sizeof(glm::vec4), &colorUpdated);
        }
        ImGui::SameLine();
        if(!_classIsSet[i]) {
          std::string label = "Track##" + std::to_string(i);
          if(ImGui::Button(label.c_str())) { setClass(i); }
        } else {
          std::string label = "Untrack##" + std::to_string(i);
          if(ImGui::Button(label.c_str())) { unsetClass(i); }
        }
      }

      if(_nSelectedNeighbors > 0) {
        ImGui::Text("No. of selected neighbours: %u", _nSelectedNeighbors);
        ImGui::SameLine();
        if(_classesSet.size() == 2) { ImGui::Text("(interclass)"); } else { ImGui::Text("(all)"); }
      } else { ImGui::Dummy(ImVec2(43.0f, 13.0f)); }
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

    // Calculate selection average and/or variance per attribute
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, i + 1, _selectionCounts[i], _buffersTextureData[i * 2]);
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().averageTexturedata(_minimizationBuffers.dataset, _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, i + 1, _selectionCounts[i], _buffersTextureData[i * 2 + 1], true, _buffersTextureData[i * 2]); // Variance
    }
    for(uint i = 0; i < 2; ++i) {
      dh::util::BufferTools::instance().difference(_buffersTextureData[i], _buffersTextureData[i+2], _params->nHighDims, _buffersTextureData[i+4]);
    }

    // Compute pairwise attribute differences if relevant texture tabs are open
    uint textureIndex = _currentTabUpper * 2 + _currentTabLower;
    if(textureIndex == 6 || textureIndex == 8) {
      glClearNamedBufferData(_buffers(BufferType::ePairwiseAttrDists), GL_R32F, GL_RED, GL_FLOAT, nullptr);
      glClearNamedBufferData(_similaritiesBuffers.neighborsSelected, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::ePairwiseAttrDiffsNeiComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      if(_classesSet.size() == 2) {
        std::pair<int, int> classes(*_classesSet.begin(), *std::next(_classesSet.begin()));
        program.template uniform<bool>("classesSet", true);
        program.template uniform<int>("classA", classes.first);
        program.template uniform<int>("classB", classes.second);
      } else {
        program.template uniform<bool>("classesSet", false);
      }

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.dataset);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimizationBuffers.labels);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.neighborsSelected);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairwiseAttrDists));

      // Dispatch shader in batches of batchSize attributes
      glDispatchCompute(ceilDiv(_params->n * _params->nHighDims, 256u), 1, 1);

      uint nSelectedNeighbors = dh::util::BufferTools::instance().reduce<uint>(_similaritiesBuffers.neighborsSelected, 0, _params->n, _minimizationBuffers.selection, -1, true, _similaritiesBuffers.layout, _similaritiesBuffers.neighbors);
      dh::util::BufferTools::instance().averageTexturedata(_buffers(BufferType::ePairwiseAttrDists), _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, nSelectedNeighbors, _buffersTextureData(TextureType::ePairwiseDiffsNei));
      glAssert();
    }

    if(textureIndex == 7 || textureIndex == 8) {
      glClearNamedBufferData(_buffers(BufferType::ePairwiseAttrDists), GL_R32F, GL_RED, GL_FLOAT, nullptr);

      auto &program = _programs(ProgramType::ePairwiseAttrDiffsAllComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("nHighDims", _params->nHighDims);
      if(_classesSet.size() == 2) {
        std::pair<int, int> classes(*_classesSet.begin(), *std::next(_classesSet.begin()));
        program.template uniform<bool>("classesSet", true);
        program.template uniform<int>("classA", classes.first);
        program.template uniform<int>("classB", classes.second);
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

      // Dispatch shader in batches of batchSize attributes
      glDispatchCompute(ceilDiv(_params->n * _params->nHighDims, 256u), 1, 1);

      uint nSelectedPairs = _selectionCounts[0] * (_selectionCounts[0] - 1);
      dh::util::BufferTools::instance().averageTexturedata(_buffers(BufferType::ePairwiseAttrDists), _params->n, _params->nHighDims, _params->imgDepth, _minimizationBuffers.selection, 1, nSelectedPairs, _buffersTextureData(TextureType::ePairwiseDiffsAll));
      glAssert();
    }

    if(textureIndex == 8) {
      dh::util::BufferTools::instance().difference(_buffersTextureData(TextureType::ePairwiseDiffsNei), _buffersTextureData(TextureType::ePairwiseDiffsAll), _params->nHighDims, _buffersTextureData(TextureType::ePairwiseDiffsDif));
    }
  }

  void AttributeRenderTask::clear() {
    for(uint i = 0; i < _buffersTextureData.size() - 3; ++i) {
      glClearNamedBufferData(_buffersTextureData[i], GL_R32F, GL_RED, GL_FLOAT, nullptr);
    }
    _nSelectedNeighbors = 0;
  }

} // dh::vis