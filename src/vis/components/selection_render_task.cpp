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
  : RenderTask(-1, "SelectionRenderTask"), 
    _isInit(false),
    _selectionRadius(5) {

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

    _isInit = true;
  }

  SelectionRenderTask::SelectionRenderTask(sne::MinimizationBuffers minimization, sne::Params params, int priority)
  : RenderTask(priority, "SelectionRenderTask"), 
    _isInit(false),
    _minimization(minimization),
    _params(params),
    _selectionRadius(5) {
    // Enable/disable render task by default
    enable = DH_VIS_EMBEDDING_INIT;

    // Initialize shader program
    {
      // if constexpr (D == 2) {
      //   _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/2D/selection.vert"));
      //   _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/2D/selection.frag"));
      // } else if constexpr (D == 3) {
      //   _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/3D/selection.vert"));
      //   _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/3D/selection.frag"));
      // }
      _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/2D/selection.vert"));
      _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/2D/selection.frag"));
      _program.link();
      glAssert();
    }

    _isInit = true;
  }

  SelectionRenderTask::~SelectionRenderTask() {
    if (_isInit) {
      // ...
    }
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

    _program.bind();

    // Set uniforms
    _program.template uniform<float, 2>("cursorPosition", _cursorPosition);
    _program.template uniform<int>("selectionRadius", _selectionRadius);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelsHandle);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr, _params.n);
  }

  void SelectionRenderTask::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Selection render settings")) {
      ImGui::Spacing();
      ImGui::SliderInt("Selection radius", &_selectionRadius, 1, 100);
      ImGui::Spacing();
    }
  }

} // dh::vis