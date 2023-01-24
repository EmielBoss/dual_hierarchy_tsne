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
#include "dh/util/gl/error.hpp"
#include "dh/vis/components/selection_render_task.hpp"

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
  
  SelectionRenderTask::SelectionRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  SelectionRenderTask::SelectionRenderTask(std::array<GLuint, 7> texturedataBuffers, GLuint attributeWeightsBuffer, sne::Params* params, int priority, const float* dataPtr)
  : RenderTask(priority, "SelectionRenderTask"),
    _isInit(false),
    _texturedataBuffers(texturedataBuffers),
    _attributeWeightsBuffer(attributeWeightsBuffer),
    _params(params),
    _selectLabeledOnly(false),
    _selectionRadiusRel(0.03),
    _selectionCounts(2, 0),
    _mousePosScreen({0.0, 0.0}),
    _draggedTexel(-1),
    _buttonPressed(0),
    _selectedDatapoint(0),
    _attributeWeight(0.f),
    _texelBrushRadius(1),
    _similarityWeight(2.f),
    _autoselectPercentage(0.1f),
    _textureTabOpened(0),
    _plotError(true),
    _currentTypeTab(0) {

    // Initialize shader program
    {
      _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/selection.vert"));
      _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/selection.frag"));
      _program.link();
      glAssert();
    }

    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::ePositions), quadPositions.size() * sizeof(glm::vec2), quadPositions.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eElements), quadElements.size() * sizeof(uint), quadElements.data(), 0);
      glAssert();
    }

    // Initialize vertex array object
    {
      glCreateVertexArrays(1, &_vaoHandle);

      // Specify vertex buffers and element buffer
      glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(glm::vec2));   // Quad positions
      glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));                            // Quad elements/indices

      // Embedding positions advance once for the full set of vertices drawn
      glVertexArrayBindingDivisor(_vaoHandle, 0, 0);
      
      // Specify vertex array data organization
      glVertexArrayAttribFormat(_vaoHandle, 0, 2, GL_FLOAT, GL_FALSE, 0);

      // Other VAO properties
      glEnableVertexArrayAttrib(_vaoHandle, 0);
      glVertexArrayAttribBinding(_vaoHandle, 0, 0);
      
      glAssert();
    }

    if(_params->imageDataset) {
      glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
      for(uint i = 0; i < _textures.size() - 1; ++i) {
        glTextureParameteri(_textures[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(_textures[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        GLenum formatInternal = _params->imgDepth == 1 ? GL_R8 : GL_RGB8;
        glTextureStorage2D(_textures[i], 1, formatInternal, _params->imgWidth, _params->imgHeight);
      }
      glTextureParameteri(_textures[_textures.size()-1], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures[_textures.size()-1], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(_textures[_textures.size()-1], 1, GL_RGBA8, _params->imgWidth, _params->imgHeight);
    }


    _isInit = true;
  }

  SelectionRenderTask::~SelectionRenderTask() {
    glDeleteTextures(_textures.size(), _textures.data());
  }

  SelectionRenderTask::SelectionRenderTask(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
  }

  SelectionRenderTask& SelectionRenderTask::operator=(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void SelectionRenderTask::render(glm::mat4 model_view, glm::mat4 proj) {
    if (!enable) {
      return;
    }

    _program.bind();

    int resolution[4]; // First two are offsets (and likely 0)
    glGetIntegerv(GL_VIEWPORT, resolution); // Ask OpenGL for the viewport dimensions, since we don't have easy acces to a _windowHandle here
    int selectionRadiusScreen = _selectionRadiusRel * (float) resolution[3] * 0.915244; // The embedding box covers 91.5244% of the screen vertically

    // Set uniforms
    _program.template uniform<float, 2>("mousePosScreen", _mousePosScreen);
    _program.template uniform<int>("selectionRadiusScreen", selectionRadiusScreen);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElements(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr);
  }

  void SelectionRenderTask::drawImGuiComponent() {
    // Copy texture data to textures
    if(_params->imageDataset) {
      for(uint i = 0; i < _textures.size() - 1; ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _texturedataBuffers[i]);
        GLenum format = _params->imgDepth == 1 ? GL_RED : GL_RGB;
        glTextureSubImage2D(_textures[i], 0, 0, 0, _params->imgWidth, _params->imgHeight, format, GL_FLOAT, 0);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _texturedataBuffers[_textures.size()-1]);
      glTextureSubImage2D(_textures[_textures.size()-1], 0, 0, 0, _params->imgWidth, _params->imgHeight, GL_RGBA, GL_FLOAT, 0);
    }
    glAssert();

    if (ImGui::CollapsingHeader("Selection render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (_canDrawLabels) {
        ImGui::Text("Selection mode:");
        if (ImGui::RadioButton("All", _selectLabeledOnly==false)) { _selectLabeledOnly = false; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Only labeled", _selectLabeledOnly==true)) { _selectLabeledOnly = true; }
      }

      ImGui::InputInt("Select individual datapoint", &_selectedDatapoint, 1, 100, ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);

      _buttonPressed = 0;
      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
      ImGui::Text("Sim. weight:"); ImGui::SameLine(); ImGui::SliderFloat("%", &_similarityWeight, 0.0f, _params->maxSimilarityWeight);
      if(ImGui::SameLine(); ImGui::Button("Apply")) { _buttonPressed = 1; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities of the selected datapoints with the specified weight."); ImGui::EndTooltip(); }
      if(_selectionCounts[1] > 0) {
        if(ImGui::SameLine(); ImGui::Button("Inter")) { _buttonPressed = 10; }
        if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities between datapoints from different selections with the specified weight."); ImGui::EndTooltip(); }
      } else { ImGui::SameLine(); ImGui::Dummy(ImVec2(43.0f, 19.0f)); }

      ImGui::Text("Select:");
      if(ImGui::SameLine(); ImGui::Button("all")) { _buttonPressed = 20; }
      if(ImGui::SameLine(); ImGui::Button("inverse")) { _buttonPressed = 30; }

      ImGui::Text("Selection size (primary): %i", _selectionCounts[0]);
      ImGui::Text("Selection size (secondary): %i", _selectionCounts[1]);
    }

    if (ImGui::BeginTabBar("Selection tabs")) {
      if (ImGui::BeginTabItem(_selectionCounts[1] > 0 ? "Selection primary" : "Selection")) {
        _currentSelectionTab = 0;
        ImGui::EndTabItem();
      }

      if(_selectionCounts[1] > 0) {
        if (ImGui::BeginTabItem("Selection secondary")) {
          _currentSelectionTab = 1;
          ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Selection diff")) {
          _currentSelectionTab = 2;
          ImGui::EndTabItem();
        }
      }
      ImGui::EndTabBar();
    }

    _draggedTexel = -1;

    if (ImGui::BeginTabBar("Selection attributes type tabs")) {
      drawImGuiTab(_currentSelectionTab, 0, "Average");
      drawImGuiTab(_currentSelectionTab, 1, "Variance");
      ImGui::EndTabBar();
    }
  }

  void SelectionRenderTask::drawImGuiTab(uint selectionIndex, uint typeIndex, const char* text) {
    if (ImGui::BeginTabItem(text)) {
      if(_params->imageDataset) {
        drawImGuiTexture(selectionIndex * 2 + typeIndex);
        drawImGuiTextureControls();
      } else {
        drawImPlotBarPlot(selectionIndex);
      }
      _currentTypeTab = typeIndex;
      ImGui::EndTabItem();
    }
  }

  // Gets a specified location in a specified buffer
  float SelectionRenderTask::getBufferValue(GLuint buffer, int index) {
    float value;
    glGetNamedBufferSubData(buffer, index * sizeof(float), sizeof(float), &value);
    return value;
  }

  void SelectionRenderTask::drawImGuiTexture(uint index) {
    _textureTabOpened = index;

    ImGui::Spacing();
    ImGui::ImageButton((void*)(intptr_t)_textures[index], ImVec2(256, 256), ImVec2(0,0), ImVec2(1,1), 0);
    // ImGui::ImageButton((void*)(intptr_t)_textures[6], ImVec2(256, 256), ImVec2(0,0), ImVec2(1,1), 0);

    if(ImGui::IsItemHovered()) {
      ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures[6], ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
      _hoveringTexture = true;
      uint teXel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 256 * _params->imgWidth;
      uint teYel = (ImGui::GetMousePos().y - ImGui::GetItemRectMin().y) / 256 * _params->imgHeight;
      uint hoveredTexel = teYel * _params->imgWidth + teXel;

      ImGui::BeginTooltip();
      ImGui::Text("Attribute: #%d", hoveredTexel);
      ImGui::Text("Weight: %0.2f", getBufferValue(_attributeWeightsBuffer, hoveredTexel));
      std::array<const char*, 6> prompts = {"Mean: %0.2f", "Variance: %0.2f", "Mean: %0.2f", "Variance: %0.2f", "Difference in mean: %0.2f", "Difference in variance: %0.2f"};
      float texelValue = getBufferValue(_texturedataBuffers[index], hoveredTexel * _params->imgDepth);
      ImGui::Text(prompts[index], texelValue);
      ImGui::EndTooltip();

      if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
      else { _draggedTexel = -1; }
    } else {
      _hoveringTexture = false;
      _draggedTexel = -1;
    }
    ImGui::SameLine(); ImGui::VSliderFloat("##v", ImVec2(40, 256), &_attributeWeight, 0.0f, _params->maxAttributeWeight, "Attr\nWght\n%.2f");
    ImGui::SameLine(); ImGui::VSliderInt("##i", ImVec2(40, 256), &_texelBrushRadius, 0, 10, "Brsh\nSize\n%i");
    glAssert();
  }

  void SelectionRenderTask::drawImGuiTextureControls() {
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
    if(ImGui::Button("Autoweigh")) { _buttonPressed = 15; }
    ImGui::SameLine(); ImGui::Text("top"); ImGui::SameLine(); ImGui::SliderFloat("\% of attribs", &_autoselectPercentage, 0.0f, 1.f);
    if(                   ImGui::Button("Recalc similarities")) { _buttonPressed = 2; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Reset similarities")) { _buttonPressed = 3; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Reinstates the original similarities calculated from the dataset."); ImGui::EndTooltip(); }
    if(                          ImGui::Button("Clear weights")) { _buttonPressed = 4; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Invert weights")) { _buttonPressed = 5; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute weights."); ImGui::EndTooltip(); }
    if(ImGui::SameLine(); ImGui::Button("Refine weights")) { _buttonPressed = 6; }
    if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Refines current attribute weights."); ImGui::EndTooltip(); }
  }

  void SelectionRenderTask::drawImPlotBarPlot(uint selectionIndex) {
    ImPlot::SetNextAxesLimits(0.f, (float) _params->nHighDims, 0.f, 1.f);
    if (ImPlot::BeginPlot("##")) {
      std::vector<float> xs(_params->nHighDims);
      std::iota(xs.begin(), xs.end(), 0); // Fills xs with 0..nHighDims-1
      std::vector<float> ys(_params->nHighDims);
      glGetNamedBufferSubData(_texturedataBuffers[selectionIndex * 2], 0, _params->nHighDims * sizeof(float), ys.data());
      std::vector<float> errs(_params->nHighDims);
      glGetNamedBufferSubData(_texturedataBuffers[selectionIndex * 2 + 1], 0, _params->nHighDims * sizeof(float), errs.data());

      ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_NoDecorations); // ImPlot 0.14 or later

      ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);
      if(_plotError) {
        ImPlot::SetNextErrorBarStyle(ImPlot::GetColormapColor(1), 0.1f, 0.1f);
        ImPlot::PlotErrorBars("Average", xs.data(), ys.data(), errs.data(), _params->nHighDims);
        ImPlot::PlotBars("Average", xs.data(), ys.data(), _params->nHighDims, 1.f);
      }
      ImPlot::EndPlot();
    }
    ImGui::Checkbox("Plot error bars", &_plotError);
  }

  // Draws key command list
  void SelectionRenderTask::drawImGuiComponentSecondary() {
    if (ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Scroll: brush size");
      ImGui::Text("Left mouse button (hold): move and fix selection"); if(_input.mouseLeft) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Right mouse button (hold): select datapoints"); if(_input.mouseRight) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("S (hold): secondary select"); if(_input.s) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("D (press): deselect"); if(_input.d) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("F (press): unfix datapoints"); if(_input.f) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("E (hold): exaggerate"); if(_input.e) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("R (press): reinitialize (fixed seed)"); if(_input.r) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Alt + R (press): reinitialize (random)"); if(_input.r && _input.alt) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Alt + Num (press): reinitialize (num seed)"); if(_input.r && _input.num >= 0) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Space (hold): pause minimization"); if(_input.space) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Delete (press): remove selected datapoints"); if(_input.del) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Insert (press): restore removed datapoints"); if(_input.ins) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Num: undefined"); if(_input.num >= 0) { ImGui::SameLine(); ImGui::Text("%i", _input.num); }
    }
  }

} // dh::vis