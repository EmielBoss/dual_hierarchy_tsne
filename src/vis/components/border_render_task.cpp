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
#include <random>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
#include "dh/util/gl/error.hpp"
#include "dh/vis/components/border_render_task.hpp"

namespace dh::vis {
  // Rectangle vertex position data
  constexpr float quadPositions[] = {
    0.f, 0.f,  // 0
    1.0f, 0.f,   // 1
    1.0f, 1.0f,    // 2
    0.f, 1.0f    // 3
  };
  
  template <uint D>
  BorderRenderTask<D>::BorderRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  BorderRenderTask<D>::BorderRenderTask(sne::MinimizationBuffers minimization, sne::Params params, int priority)
  : RenderTask(priority, "BorderRenderTask"), 
    _isInit(false),
    _minimization(minimization),
    _params(params) {
    // Enable/disable render task by default
    enable = DH_VIS_EMBEDDING_INIT;

    // Initialize shader program
    {
      if constexpr (D == 2) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/border/2D/border.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/border/2D/border.frag"));
      } else if constexpr (D == 3) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/border/3D/border.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/border/3D/border.frag"));
      }
      _program.link();
      glAssert();
    }

    // Initialize vertex array object
    {
      glGenVertexArrays(1, &_vaoHandle);
      glBindVertexArray(_vaoHandle);

      // Specify vertex buffers/VAO
      glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(glm::vec2));

      // Create positions buffer/VBO
      glGenBuffers(1, &_buffers(BufferType::ePositions));
      glBindBuffer(GL_ARRAY_BUFFER, _buffers(BufferType::ePositions));
      glBufferData(GL_ARRAY_BUFFER, sizeof(quadPositions), quadPositions, GL_DYNAMIC_DRAW);
      
      // Specify vertex array data organization
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
      glEnableVertexAttribArray(0);
      
      glAssert();
    }

    _isInit = true;
  }

  template <uint D>
  BorderRenderTask<D>::~BorderRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  template <uint D>
  BorderRenderTask<D>::BorderRenderTask(BorderRenderTask&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  BorderRenderTask<D>& BorderRenderTask<D>::operator=(BorderRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void BorderRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {
    if (!enable) {
      return;
    }

    _program.bind();

    // Set uniforms
    _program.template uniform<float, 4, 4>("model_view", model_view);
    _program.template uniform<float, 4, 4>("proj", proj);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawArrays(GL_LINE_LOOP, 0, 4);
  }

  template <uint D>
  void BorderRenderTask<D>::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Border render settings")) {
      ImGui::Spacing();
      // ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.0f, 1.0f);
      ImGui::Spacing();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class BorderRenderTask<2>;
  template class BorderRenderTask<3>;
} // dh::vis