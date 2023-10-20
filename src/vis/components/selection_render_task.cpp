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

  SelectionRenderTask::SelectionRenderTask(sne::Params* params, int priority)
  : RenderTask(priority, "SelectionRenderTask"),
    _isInit(false),
    _params(params),
    _selectLabeledOnly(false),
    _selectionRadiusRel(0.03),
    _selectionCounts(2, 0),
    _mousePosScreen({0.0, 0.0}),
    _buttonPressed(0),
    _selectedDatapoint(0),
    _similarityWeight(1.f) {

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
  }

  SelectionRenderTask::SelectionRenderTask(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
  }

  SelectionRenderTask& SelectionRenderTask::operator=(SelectionRenderTask&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void SelectionRenderTask::render(glm::mat4 model_view, glm::mat4 proj) {
    if (!enabled) {
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
    if (ImGui::CollapsingHeader("Selection render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Selection mode:");
      ImGui::SameLine();
      if (ImGui::RadioButton("All", _selectLabeledOnly==false)) { _selectLabeledOnly = false; }
      ImGui::SameLine();
      if (ImGui::RadioButton("Only labeled", _selectLabeledOnly==true)) { _selectLabeledOnly = true; }

      ImGui::InputInt("Select individual datapoint", &_selectedDatapoint, 1, 100, ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue);

      _buttonPressed = 0;
      ImGui::Text("Sim. weight:");
      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.3f);
      ImGui::SameLine(); ImGui::SliderFloat("%", &_similarityWeight, 0.0f, _params->maxSimilarityWeight);
      if(ImGui::SameLine(); ImGui::Button("Apply")) { _buttonPressed = 1; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities of the selected datapoints with the specified weight."); ImGui::EndTooltip(); }
      if(_selectionCounts[0] > 0 && _selectionCounts[1] > 0) {
        if(ImGui::SameLine(); ImGui::Button("Inter")) { _buttonPressed = 10; }
        if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Weight the similarities between datapoints from different selections with the specified weight."); ImGui::EndTooltip(); }
      } else { ImGui::SameLine(); ImGui::Dummy(ImVec2(43.0f, 19.0f)); }

      ImGui::Text("Select:");
      if(ImGui::SameLine(); ImGui::Button("all")) { _buttonPressed = 20; }
      if(ImGui::SameLine(); ImGui::Button("inverse")) { _buttonPressed = 30; }

      ImGui::Text("Selection size (primary): %i", _selectionCounts[0]);
      ImGui::Text("Selection size (secondary): %i", _selectionCounts[1]);
      
      if(_selectionCounts[0] > 0 && _selectionCounts[1] > 0) {
        if(ImGui::Button("Fuse (inter)")) { _buttonPressed = 11; }
        if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Add similarities between datapoints from different selections where those don't exist."); ImGui::EndTooltip(); }
      } else {
        if(ImGui::Button("Fuse (intra)")) { _buttonPressed = 12; }
        if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Add similarities between all pairs of datapoints in the primary selection where those don't exist."); ImGui::EndTooltip(); }
      }
      if(ImGui::SameLine(); ImGui::Button("Assert")) { _buttonPressed = 13; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Checks if the kNN graph is still valid."); ImGui::EndTooltip(); }
    }
  }

  // Draws key command list
  void SelectionRenderTask::drawImGuiComponentSecondary() {
    if (ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Scroll: brush size");
      ImGui::Text("Left mouse button (hold): select datapoints"); if(_input.mouseLeft) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Middle mouse button (hold): deselect datapoints"); if(_input.mouseMiddle) { ImGui::SameLine(); ImGui::Text(" <"); }
      ImGui::Text("Right mouse button (hold): move and fix selection"); if(_input.mouseRight) { ImGui::SameLine(); ImGui::Text(" <"); }
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