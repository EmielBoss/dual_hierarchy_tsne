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
    _selectionRadius(30),
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

    // Initialize textures and framebuffer and all other shit for averaging selected images
    if(_params.datapointsAreImages) {
      _textures = std::vector<GLuint>(_params.n);
      for(uint i = 0; i < _params.n; ++i) {
        glCreateTextures(GL_TEXTURE_2D, 1, &_textures[i]);
        glTextureParameteri(_textures[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(_textures[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureStorage2D(_textures[i], 1, GL_R8, _params.imgWidth, _params.imgHeight);
        std::vector<float> data(_params.imgWidth * _params.imgHeight);
        for(uint p = 0; p < _params.imgWidth * _params.imgHeight; ++p) { data[p] = *(dataPtr + i * _params.nHighDims + p) / 255.0f; }
        glTextureSubImage2D(_textures[i], 0, 0, 0, _params.imgWidth, _params.imgHeight, GL_RED,  GL_FLOAT, data.data());
      }

      glCreateTextures(GL_TEXTURE_2D, 1, &_averageSelectionTexture);
      glTextureParameteri(_averageSelectionTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_averageSelectionTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(_averageSelectionTexture, 1, GL_R8, _params.imgWidth, _params.imgHeight);

      glCreateFramebuffers(1, &_averageSelectionFramebuffer);
      glCreateRenderbuffers(1, &_averageSelectionRenderbuffer);
      glNamedRenderbufferStorage(_averageSelectionRenderbuffer, GL_R8, _params.imgWidth, _params.imgWidth);
      glNamedFramebufferRenderbuffer(_averageSelectionFramebuffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _averageSelectionRenderbuffer);
      // glNamedFramebufferTexture(_averageSelectionFramebuffer, GL_COLOR_ATTACHMENT0, _averageSelectionTexture, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0); // Bind default framebuffer again
      glAssert();

      // Specify vertex buffers/VAO
      glCreateVertexArrays(1, &_averageSelectionVAO);
      uint vbo, ebo;
      glCreateBuffers(1, &vbo);
      glNamedBufferStorage(vbo, quadPositions.size() * sizeof(glm::vec2), quadPositions.data(), 0);
      glVertexArrayVertexBuffer(_averageSelectionVAO, 0, vbo, 0, sizeof(glm::vec2));
      glEnableVertexArrayAttrib(_averageSelectionVAO, 0);
      glVertexArrayAttribBinding(_averageSelectionVAO, 0, 0);

      glCreateBuffers(1, &ebo);
      glNamedBufferStorage(ebo, quadElements.size() * sizeof(uint), quadElements.data(), 0);
      glVertexArrayElementBuffer(_averageSelectionVAO, ebo);
      glAssert();

      _averageSelectionProgram.addShader(util::GLShaderType::eVertex, rsrc::get("vis/selection/average_selection_texture.vert"));
      _averageSelectionProgram.addShader(util::GLShaderType::eFragment, rsrc::get("vis/selection/average_selection_texture.frag"));
      _averageSelectionProgram.link();
      glAssert();
    }

    _isInit = true;
  }

  SelectionRenderTask::~SelectionRenderTask() {
    if (_isInit) {
      glDeleteTextures(1, &_averageSelectionTexture);
      glDeleteTextures(_params.n, _textures.data());
      glDeleteFramebuffers(1, &_averageSelectionFramebuffer);
      glDeleteRenderbuffers(1, &_averageSelectionRenderbuffer);
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

    _canDrawLabels = (labelsHandle > 0); // Used for showing selection mode options

    _program.bind();

    // Set uniforms
    _program.template uniform<float, 2>("cursorPosition", _mousePosition);
    _program.template uniform<int>("selectionRadius", _selectionRadius);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElements(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr);
  }

  void SelectionRenderTask::averageSelectedImages() {
    if (!_isInit) {
      return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, _averageSelectionFramebuffer);
    glViewport(0, 0, _params.imgWidth, _params.imgHeight);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    std::vector<uint> selectedNewly(_params.n);

    // Ideally I want to take the else path, which is much more efficient but for some reason only works for the first ~1000 selected datapoints
    bool inefficientButAtLeastItWorks = true;
    if(inefficientButAtLeastItWorks) {
      glGetNamedBufferSubData(_minimizationBuffers.selected, 0, _params.n * sizeof(uint), selectedNewly.data());
      glClearTexImage(_averageSelectionTexture, 0, GL_RED, GL_FLOAT, nullptr);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(0x00004000);
      _averagedSelectionCount = 0;
    } else {
      glGetNamedBufferSubData(_minimizationBuffers.selectedNewly, 0, _params.n * sizeof(uint), selectedNewly.data());
    }

    _averageSelectionProgram.bind();
    glBindVertexArray(_averageSelectionVAO);

    for(uint i = 0; i < _params.n; ++i) {
      if(!selectedNewly[i]) { continue; }
      _averageSelectionProgram.template uniform<uint>("count", ++_averagedSelectionCount);
      glBindTexture(GL_TEXTURE_2D, _textures[i]);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    // Copy renderbuffer to texture
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glCopyTextureSubImage2D(_averageSelectionTexture, 0, 0, 0, 0, 0, _params.imgWidth, _params.imgHeight);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glCopyImageSubData(_textures[6], GL_TEXTURE_2D, 0, 0, 0, 0,
    //               _averageSelectionTexture, GL_TEXTURE_2D, 0, 0, 0, 0,
    //               _params.imgWidth, _params.imgHeight, 1);
    glAssert();
  }

  void SelectionRenderTask::clearSelection(bool datapointsAreImages) {
      glClearNamedBufferData(_minimizationBuffers.selected, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      if(datapointsAreImages) {
        glClearTexImage(_averageSelectionTexture, 0,  GL_RED, GL_FLOAT, nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER, _averageSelectionFramebuffer);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(0x00004000); // That's COLOR_BIT_BUFFER, but I can't access that here for some reason
        _averagedSelectionCount = 0;
      }
  }

  void SelectionRenderTask::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Selection render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Spacing();
      ImGui::SliderInt("Selection radius", &_selectionRadius, 1, 1000);
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
        ImGui::Image((void*)(intptr_t)_averageSelectionTexture, ImVec2(256, 256));
        ImGui::Spacing();
      }
    }
  }

} // dh::vis