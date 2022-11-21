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
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
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

  SelectionRenderTask::SelectionRenderTask(std::array<GLuint, 7> textures, std::array<GLuint, 7> texturedataBuffers, GLuint attributeWeightsBuffer, sne::Params params, int priority, const float* dataPtr)
  : RenderTask(priority, "SelectionRenderTask"),
    _isInit(false),
    _textures(textures),
    _texturedataBuffers(texturedataBuffers),
    _attributeWeightsBuffer(attributeWeightsBuffer),
    _params(params),
    _selectLabeledOnly(false),
    _selectionRadiusRel(0.03),
    _selectionCounts(2, 0),
    _mousePosScreen({0.0, 0.0}),
    _draggedTexel(-1),
    _buttonPressed(0),
    _selectAll(false),
    _selectedDatapoint(0),
    _attributeWeight(0.f),
    _texelBrushRadius(1),
    _similarityWeight(2.f),
    _autoselectPercentage(0.1f),
    _textureTabOpened(0),
    _perplexity(params.perplexity),
    _k((int) _params.k) {

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

    _isInit = true;
  }

  SelectionRenderTask::~SelectionRenderTask() {
    // ...
  }

  SelectionRenderTask::SelectionRenderTask(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
  }

  SelectionRenderTask& SelectionRenderTask::operator=(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void SelectionRenderTask::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {
    if (!enable) {
      return;
    }

    _canDrawLabels = (labelsHandle > 0); // Used for showing selection mode options

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
    if (ImGui::CollapsingHeader("Selection render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (_canDrawLabels) {
        ImGui::Text("Selection mode:");
        if (ImGui::RadioButton("All", _selectLabeledOnly==false)) { _selectLabeledOnly = false; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Only labeled", _selectLabeledOnly==true)) { _selectLabeledOnly = true; }
      }

      ImGui::InputInt("Select individual datapoint", &_selectedDatapoint, 1, 100, ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);

      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
      ImGui::Text("Similarity weight:"); ImGui::SameLine(); ImGui::SliderFloat("%", &_similarityWeight, 0.0f, _params.maxSimilarityWeight);

      ImGui::Text("No. of sel. points: %i", _selectionCounts[0]);
      if(ImGui::SameLine(); ImGui::Button("Select all")) { _selectAll = true; }
      else { _selectAll = false; }
      _buttonPressed = 0;
      if(ImGui::SameLine(); ImGui::Button("Apply weight")) { _buttonPressed = 1; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities of the selected datapoints with the specified weight."); ImGui::EndTooltip(); }
      ImGui::Spacing();

      if(_selectionCounts[1] > 0) {
        ImGui::Text("No. of sel. points (secondary): %i", _selectionCounts[1]);
        if(ImGui::SameLine(); ImGui::Button("Apply  weight")) { _buttonPressed = 10; }
        if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities between datapoints from different selections with the specified weight."); ImGui::EndTooltip(); }
      } else {
        ImGui::Dummy(ImVec2(0.0f, 19.0f));
      }
    }

    if(!_params.imageDataset) { return; }

    if (ImGui::CollapsingHeader("Attribute weighing settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      _draggedTexel = -1;
      if (ImGui::BeginTabBar("Selection textures", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Avg")) {
            drawImGuiImageButton(0);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Var")) {
            drawImGuiImageButton(1);
            ImGui::EndTabItem();
        }
        if(_selectionCounts[1] > 0) {
          if (ImGui::BeginTabItem("Avg 2")) {
              drawImGuiImageButton(2);
              ImGui::EndTabItem();
          }
          if (ImGui::BeginTabItem("Var 2")) {
              drawImGuiImageButton(3);
              ImGui::EndTabItem();
          }
          if (ImGui::BeginTabItem("Avg diff")) {
              drawImGuiImageButton(4);
              ImGui::EndTabItem();
          }
          if (ImGui::BeginTabItem("Var diff")) {
              drawImGuiImageButton(5);
              ImGui::EndTabItem();
          }
        }
        ImGui::EndTabBar();
      }

      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5f);
      if(ImGui::Button("Autoweigh")) { _buttonPressed = 15; }
      ImGui::SameLine(); ImGui::Text("top"); ImGui::SameLine(); ImGui::SliderFloat("\% of attribs", &_autoselectPercentage, 0.0f, 1.f);
      if(                   ImGui::Button("Recalc similarities")) { _buttonPressed = 2; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Recalculates similarities of the selected datapoints by weighting the selected attributes."); ImGui::EndTooltip(); }
      if(ImGui::SameLine(); ImGui::Button("Reset similarities")) { _buttonPressed = 3; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Reinstates the original similarities calculated from the dataset."); ImGui::EndTooltip(); }
      if(                          ImGui::Button("Clear selection")) { _buttonPressed = 4; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Clears current attribute selection."); ImGui::EndTooltip(); }
      if(ImGui::SameLine(); ImGui::Button("Invert selection")) { _buttonPressed = 5; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Inverts current attribute selection."); ImGui::EndTooltip(); }
      if(ImGui::SameLine(); ImGui::Button("Refine selection")) { _buttonPressed = 6; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Refines current attribute selection."); ImGui::EndTooltip(); }
      // ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
      // ImGui::SliderFloat("Perpl.", &_perplexity, 1.0f, 100.f);
      // if(ImGui::IsItemHovered() && ImGui::IsItemActive()) { _k = (int) std::min(_params.kMax, 3 * (uint)(_perplexity) + 1); }
      // ImGui::SameLine(); ImGui::SliderInt("k", &_k, 2, _params.kMax);
      // ImGui::PopItemWidth();
    }
  }

  // Gets a specified location in a specified buffer
  float SelectionRenderTask::getBufferValue(GLuint buffer, int index) {
    float value;
    glGetNamedBufferSubData(buffer, index * sizeof(float), sizeof(float), &value);
    return value;
  }

  void SelectionRenderTask::drawImGuiImageButton(uint index) {
    _textureTabOpened = index;

    ImGui::Spacing();
    ImGui::ImageButton((void*)(intptr_t)_textures[index], ImVec2(256, 256), ImVec2(0,0), ImVec2(1,1), 0);
    // ImGui::ImageButton((void*)(intptr_t)_textures[6], ImVec2(256, 256), ImVec2(0,0), ImVec2(1,1), 0);

    if(ImGui::IsItemHovered()) {
      ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)_textures[6], ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), ImVec2(0,0), ImVec2(1,1));
      _hoveringTexture = true;
      uint teXel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) / 256 * _params.imgWidth;
      uint teYel = (ImGui::GetMousePos().y - ImGui::GetItemRectMin().y) / 256 * _params.imgHeight;
      uint hoveredTexel = teYel * _params.imgWidth + teXel;

      ImGui::BeginTooltip();
      ImGui::Text("Attribute: #%d", hoveredTexel);
      ImGui::Text("Weight: %0.2f", getBufferValue(_attributeWeightsBuffer, hoveredTexel));
      std::array<const char*, 6> prompts = {"Mean: %0.2f", "Variance: %0.2f", "Mean: %0.2f", "Variance: %0.2f", "Difference in mean: %0.2f", "Difference in variance: %0.2f"};
      bool isVariance = index % 2 == 1;
      float texelValue = getBufferValue(_texturedataBuffers[index], hoveredTexel * 4);
      if(isVariance) { texelValue /= 2.f; } // Remove the texture boosting in order to print actual variance
      ImGui::Text(prompts[index], texelValue);
      if(isVariance) { ImGui::Text("Texture color value is x2'ed for better visibility."); }
      ImGui::EndTooltip();

      if(ImGui::IsAnyMouseDown()) { _draggedTexel = hoveredTexel; }
      else { _draggedTexel = -1; }
    } else {
      _hoveringTexture = false;
      _draggedTexel = -1;
    }
    ImGui::SameLine(); ImGui::VSliderFloat("##v", ImVec2(40, 256), &_attributeWeight, 0.0f, _params.maxAttributeWeight, "Attr\nWght\n%.2f");
    ImGui::SameLine(); ImGui::VSliderInt("##i", ImVec2(40, 256), &_texelBrushRadius, 0, 10, "Brsh\nSize\n%i");
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