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
#include "dh/vis/components/axes_render_task.hpp"

namespace dh::vis {
  // Rectangle vertex position data
  constexpr float quadPositions2D[] = {
    0.0f, 0.0f,         // 0
    1.0f, 0.0f,         // 1
    1.0f, 1.0f,         // 2
    0.0f, 1.0f          // 3
  };

  constexpr float axesPositions3D[] = {
    0.0f, 0.0f, 0.0f,   // 0
    1.0f, 0.0f, 0.0f,   // 1
    0.0f, 0.0f, 0.0f,   // 2
    0.0f, 1.0f, 0.0f,   // 3
    0.0f, 0.0f, 0.0f,   // 4
    0.0f, 0.0f, 1.0f,   // 5
  };

  constexpr float axesColors3D[] = {
    1.0f, 0.0f, 0.0f,   // 0
    1.0f, 0.0f, 0.0f,   // 1
    0.0f, 1.0f, 0.0f,   // 2
    0.0f, 1.0f, 0.0f,   // 3
    0.0f, 0.0f, 1.0f,   // 4
    0.0f, 0.0f, 1.0f,   // 5
  };
  
  template <uint D>
  AxesRenderTask<D>::AxesRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  AxesRenderTask<D>::AxesRenderTask(sne::MinimizationBuffers minimization, sne::Params params, std::vector<char> axisMapping, int selectedAxistype, int priority)
  : RenderTask(priority, "AxesRenderTask"), 
    _isInit(false),
    _minimization(minimization),
    _params(params),
    _axisMapping(axisMapping),
    _axisMappingPrev(axisMapping),
    _selectedAxistype(selectedAxistype),
    _selectedIndex(0) {
    // Enable/disable render task by default
    enable = DH_VIS_EMBEDDING_INIT;

    // Initialize shader program
    {
      if constexpr (D == 2) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/axes/2D/axes.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/axes/2D/axes.frag"));
      } else if constexpr (D == 3) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/axes/3D/axes.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/axes/3D/axes.frag"));
      }
      _program.link();
      glAssert();
    }

    // Initialize vertex array object
    {
      glGenVertexArrays(1, &_vaoHandle);
      glBindVertexArray(_vaoHandle);

      // Create positions buffer/VBO
      glGenBuffers(1, &_vboHandlePositions);
      glBindBuffer(GL_ARRAY_BUFFER, _vboHandlePositions);
      if constexpr (D == 2) { glBufferData(GL_ARRAY_BUFFER, sizeof(quadPositions2D), quadPositions2D, GL_DYNAMIC_DRAW); } else
      if constexpr (D == 3) { glBufferData(GL_ARRAY_BUFFER, sizeof(axesPositions3D), axesPositions3D, GL_DYNAMIC_DRAW); }
      if constexpr (D == 3) { 
        glGenBuffers(1, &_vboHandleColors);
        glBindBuffer(GL_ARRAY_BUFFER, _vboHandleColors);
        glBufferData(GL_ARRAY_BUFFER, sizeof(axesColors3D), axesColors3D, GL_DYNAMIC_DRAW);
      }

      // Specify vertex buffers/VAO
      if constexpr (D == 2) { glVertexArrayVertexBuffer(_vaoHandle, 0, _vboHandlePositions, 0, sizeof(glm::vec2)); } else
      if constexpr (D == 3) { glVertexArrayVertexBuffer(_vaoHandle, 0, _vboHandlePositions, 0, sizeof(glm::vec3)); }
      if constexpr (D == 3) {
        glBindBuffer(GL_ARRAY_BUFFER, _vboHandleColors);
        glVertexArrayVertexBuffer(_vaoHandle, 1, _vboHandleColors, 0, sizeof(glm::vec3));
      }
      
      // Specify vertex array data organization
      glBindBuffer(GL_ARRAY_BUFFER, _vboHandlePositions);
      if constexpr (D == 2) { glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0); } else
      if constexpr (D == 3) { glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); }
      if constexpr (D == 3) {
        glBindBuffer(GL_ARRAY_BUFFER, _vboHandleColors);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); }
      glBindBuffer(GL_ARRAY_BUFFER, _vboHandlePositions);
      glEnableVertexAttribArray(0);
      if constexpr (D == 3) {
        glBindBuffer(GL_ARRAY_BUFFER, _vboHandleColors);
        glEnableVertexAttribArray(1); }
      
      glAssert();
    }

    _isInit = true;
  }

  template <uint D>
  AxesRenderTask<D>::~AxesRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(1, &_vboHandlePositions);
      glDeleteBuffers(1, &_vboHandleColors);
    }
  }

  template <uint D>
  AxesRenderTask<D>::AxesRenderTask(AxesRenderTask&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  AxesRenderTask<D>& AxesRenderTask<D>::operator=(AxesRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void AxesRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {
    if (!enable) {
      return;
    }

    _program.bind();

    // Set uniforms
    _program.template uniform<float, 4, 4>("model_view", model_view);
    _program.template uniform<float, 4, 4>("proj", proj);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    if(D == 2) { glDrawArrays(GL_LINE_LOOP, 0, 4); } else
    if(D == 3) { glDrawArrays(GL_LINES, 0, 6); }
  }

  template <uint D>
  void AxesRenderTask<D>::drawImGuiComponent() {
    const char* axistypes[] = {"t-sne", "PCA", "Attribute"};
    const char axistypesAbbr[3] = {'t', 'p', 'a'};
    if (ImGui::CollapsingHeader("Axes settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Spacing();
      ImGui::Text("Z");
      ImGui::Combo("Axis type", &_selectedAxistype, axistypes, IM_ARRAYSIZE(axistypes));
      _axisMapping[2] = axistypesAbbr[_selectedAxistype];
      if(_axisMapping != _axisMappingPrev) {
        _selectedIndex = 0;
        _axisMappingPrev = _axisMapping;
      }
      if(_axisMapping[2] == 'p') {
        ImGui::SliderInt("Principal component", &_selectedIndex, 0, _params.nPCs-1);
      } else
      if (_axisMapping[2] == 'a') {
        ImGui::SliderInt("Attribute", &_selectedIndex, 0, _params.nHighDims-1);
      }
      ImGui::Spacing();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class AxesRenderTask<2>;
  template class AxesRenderTask<3>;
} // dh::vis