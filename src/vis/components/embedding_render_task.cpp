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
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
#include "dh/util/io.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/vis/components/embedding_render_task.hpp"

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
  
  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, const float* colorsPtr)
  : RenderTask(priority, "EmbeddingRenderTask"),
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _params(params),
    _colorMapping(ColorMapping::labels),
    _forceWeight(_params->k * 5.f),
    _forceWeightFalloff(0.f),
    _fixedRigidness(1.f),
    _selectLabeledOnly(false),
    _separationMode(false),
    _pointRadius(std::min(100.f / _params->n, 0.005f)),
    _pointOpacity(1.0f),
    _buttonPressed(0),
    _perplexity(params->perplexity),
    _k((int) _params->k),
    _iteration(0),
    _klDivergence(-1.f) {
    // Enable/disable render task by default
    enabled = DH_VIS_EMBEDDING_INIT;

    // Initialize shader program
    {
      if constexpr (D == 2) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/embedding/2D/embedding.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/embedding/2D/embedding.frag"));
      } else if constexpr (D == 3) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/embedding/3D/embedding.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/embedding/3D/embedding.frag"));
      }
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

    createVAO();

    generateClassColors();

    {
      std::vector<glm::vec3> colorsVec3(_params->n);
      for (uint i = 0; i < _params->n; ++i) {
        colorsVec3[i] = glm::vec3(colorsPtr[i * 3 + 0], colorsPtr[i * 3 + 1], colorsPtr[i * 3 + 2]);
      }

      glCreateBuffers(1, &_bufferPointColors);
      glNamedBufferStorage(_bufferPointColors, _params->n * sizeof(glm::vec3), colorsVec3.data(), 0);
      glAssert();
    }

    _isInit = true;
  }

  template <uint D>
  EmbeddingRenderTask<D>::~EmbeddingRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteBuffers(1, &_bufferClassColors);
    }
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(EmbeddingRenderTask&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  EmbeddingRenderTask<D>& EmbeddingRenderTask<D>::operator=(EmbeddingRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // (Re)create and initialize VAO
  template <uint D>
  void EmbeddingRenderTask<D>::createVAO() {
    glDeleteVertexArrays(1, &_vaoHandle);
    glCreateVertexArrays(1, &_vaoHandle);

    // Specify vertex buffers and element buffer
    constexpr uint embeddingStride = (D == 2) ? 2 : 4;
    glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(glm::vec2));                   // Quad positions
    glVertexArrayVertexBuffer(_vaoHandle, 1, _minimizationBuffers.embeddingRel, 0, embeddingStride * sizeof(float));    // Embedding positions
    glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));                                            // Quad elements/indices

    glVertexArrayBindingDivisor(_vaoHandle, 0, 0); // Quad positions advance once per vertex
    glVertexArrayBindingDivisor(_vaoHandle, 1, 1); // Embedding positions advance once for the full set of (6) vertices in ePositions/quadPositions drawn
    
    // Specify vertex array data organization
    constexpr uint embeddingSize = (D == 2) ? 2 : 3;
    glVertexArrayAttribFormat(_vaoHandle, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(_vaoHandle, 1, embeddingSize, GL_FLOAT, GL_FALSE, 0);

    // Other VAO properties
    glEnableVertexArrayAttrib(_vaoHandle, 0);
    glEnableVertexArrayAttrib(_vaoHandle, 1);
    glVertexArrayAttribBinding(_vaoHandle, 0, 0);
    glVertexArrayAttribBinding(_vaoHandle, 1, 1);
    
    glAssert();
  }

  template <uint D>
  void EmbeddingRenderTask<D>::generateClassColors() {
    
    // std::vector<glm::vec3> colors = {
    //   glm::vec3(16, 78, 139),
    //   glm::vec3(0, 128, 0),
    //   glm::vec3(139, 90, 43),
    //   glm::vec3(138, 43, 226),
    //   glm::vec3(255, 150, 0),
    //   glm::vec3(204, 40, 40),
    //   glm::vec3(131, 139, 131),
    //   glm::vec3(0, 205, 0),
    //   glm::vec3(0, 150, 255),
    //   glm::vec3(220, 220, 220)
    // };

    std::vector<glm::vec3> classColors = {
      glm::vec3(150, 150, 150),
      glm::vec3(230, 25, 75),
      glm::vec3(60, 180, 75),
      glm::vec3(255, 225, 25),
      glm::vec3(0, 130, 200),
      glm::vec3(245, 130, 48),
      glm::vec3(0, 128, 128),
      glm::vec3(70, 240, 240),
      glm::vec3(145, 30, 180),
      glm::vec3(250, 190, 212),
      glm::vec3(170, 110, 40),
      glm::vec3(255, 250, 200),
      glm::vec3(128, 0, 0),
      glm::vec3(170, 255, 195),
      glm::vec3(128, 128, 0),
      glm::vec3(255, 215, 180),
      glm::vec3(0, 0, 128),
      glm::vec3(128, 128, 128)
    };

    int nColorsToAdd = _params->nClasses - classColors.size();
    for(int i = 0; i < nColorsToAdd; ++i) {
      glm::vec3 newColor = classColors[i] + classColors[i+1];
      newColor /= 2.f;
      classColors.push_back(newColor);
    }

    glCreateBuffers(1, &_bufferClassColors);
    glNamedBufferStorage(_bufferClassColors, classColors.size() * sizeof(glm::vec3), classColors.data(), GL_DYNAMIC_STORAGE_BIT);
    glAssert();
  }

  template <uint D>
  void EmbeddingRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj) {

    if (!enabled) {
      return;
    }

    // Time-based effect for the secondary selection
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int looper = (std::abs(ms) / 2) % 510;
    float divisor = looper - std::max(0, looper * 2 - 510);

    _program.bind();

    // Set uniforms
    _program.template uniform<float, 4, 4>("model_view", model_view);
    _program.template uniform<float, 4, 4>("proj", proj);
    _program.template uniform<float>("pointOpacity", _pointOpacity);
    _program.template uniform<float>("pointRadius", _pointRadius);
    _program.template uniform<uint>("colorMapping", _colorMapping);
    _program.template uniform<bool>("canDrawLabels", _minimizationBuffers.labels > 0);
    _program.template uniform<bool>("selectLabeledOnly", _selectLabeledOnly);
    _program.template uniform<bool>("separationMode", _separationMode);
    _program.template uniform<float>("divisor", divisor);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.labels);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.labeled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimizationBuffers.disabled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _minimizationBuffers.fixed);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _minimizationBuffers.selection);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _minimizationBuffers.neighborhoodPreservation);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _bufferClassColors);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _bufferPointColors);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr, _params->n);
  }

  template <uint D>
  void EmbeddingRenderTask<D>::drawImGuiComponent() {
    _buttonPressed = 0;
    if (ImGui::CollapsingHeader("Embedding render settings", ImGuiTreeNodeFlags_DefaultOpen)) {

      ImGui::Text("Iteration: %u", _iteration); ImGui::SameLine(); ImGui::Text("KL: %f", _klDivergence);
      if(ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) { _buttonPressed = 2; }
      ImGui::SameLine(); ImGui::Text(" | State:");
      if(ImGui::SameLine(); ImGui::Button("Import")) { _buttonPressed = 10; }
      if(ImGui::SameLine(); ImGui::Button("Export")) { _buttonPressed = 11; }

      if (_minimizationBuffers.labels > 0) {
        ImGui::Text("Color map:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Labels", _colorMapping==ColorMapping::labels)) { _colorMapping = ColorMapping::labels; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Colors", _colorMapping==ColorMapping::colors)) { _colorMapping = ColorMapping::colors; }
        ImGui::SameLine();
        if (ImGui::RadioButton("kNN preservation", _colorMapping==ColorMapping::knnPreservation)) { _colorMapping = ColorMapping::knnPreservation; }
        ImGui::SameLine();
        if (ImGui::RadioButton("None", _colorMapping==ColorMapping::none)) { _colorMapping = ColorMapping::none; }
      }
      ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.0f, 1.0f);
      ImGui::SliderFloat("Point radius", &_pointRadius, 0.0001f, 0.005f, "%.4f");
      ImGui::Spacing();
      ImGui::SliderFloat("Force weight", &_forceWeight, 1.0f, _params->k * 10.f);
      ImGui::SliderFloat("Force weight falloff", &_forceWeightFalloff, 0.f, 1.f, "%.4f");
      ImGui::SliderFloat("Fixed point rigidness", &_fixedRigidness, 0.f, 1.f, "%.4f");
      // ImGui::Text("or set the number of clusters you see:");
      // ImGui::SliderInt("Number of apparent clusters", &_numClusters, 1, 50);
      ImGui::Spacing();

      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
      ImGui::SliderFloat("Perpl.", &_perplexity, 1.0f, 100.f);
      if(ImGui::IsItemHovered() && ImGui::IsItemActive()) { _k = (int) std::min(_params->kMax, 3 * (uint)(_perplexity) + 1); }
      ImGui::SameLine(); ImGui::SliderInt("k", &_k, 2, _params->kMax);
      if(ImGui::SameLine(); ImGui::Button("Isolate")) { _buttonPressed = 1; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Restarts minimization with only the selected datapoints and hyperparameters."); ImGui::EndTooltip(); }
      ImGui::PopItemWidth();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class EmbeddingRenderTask<2>;
  template class EmbeddingRenderTask<3>;
} // dh::vis