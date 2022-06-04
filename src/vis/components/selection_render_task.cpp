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

#include <array>
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

  SelectionRenderTask::SelectionRenderTask(sne::MinimizationBuffers minimizationBuffers, sne::Params params, int priority, const float* dataPtr)
  : RenderTask(priority, "SelectionRenderTask"),
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _params(params),
    _selectLabeledOnly(false),
    _selectionRadiusRel(0.03),
    _mousePosition({0.0, 0.0}) {

    // Initialize shader program
    {
      // if constexpr (D == 2) {
      //   _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/2D/selection.vert"));
      //   _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/2D/selection.frag"));
      // } else if constexpr (D == 3) {
      //   _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/3D/embedding.vert"));
      //   _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/3D/embedding.frag"));
      // }
      _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/2D/selection.vert"));
      _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/2D/selection.frag"));
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

    glGetIntegerv(GL_VIEWPORT, _resolution); // Ask OpenGL for the viewport dimensions, since we don't have easy acces to a _windowHandle here
    int selectionRadiusScreen = _selectionRadiusRel * (float) _resolution[3] * 0.915244; // The embedding box covers 91.5244% of the screen vertically

    // Set uniforms
    _program.template uniform<float, 2>("cursorPosition", _mousePosition);
    _program.template uniform<int>("selectionRadius", selectionRadiusScreen);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElements(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr);
  }

  void SelectionRenderTask::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Selection render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Spacing();
      ImGui::SliderFloat("Selection radius", &_selectionRadiusRel, 0.0f, 0.5f);
      ImGui::Spacing();
    }

    if (_canDrawLabels) {
      ImGui::Text("Selection mode:");
      if (ImGui::RadioButton("All", _selectLabeledOnly==false)) { _selectLabeledOnly = false; }
      if (ImGui::RadioButton("Only labeled", _selectLabeledOnly==true)) { _selectLabeledOnly = true; }
    }

    if(_params.datapointsAreImages) {
      if (ImGui::CollapsingHeader("Average selection image", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::Image((void*)(intptr_t)_minimizationBuffers.averageSelectionTexture, ImVec2(256, 256));
        ImGui::Spacing();
      }
    }
  }

} // dh::vis