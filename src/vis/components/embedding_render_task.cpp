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
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
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

  float calculateFalloff(uint n, uint k, int nClusters) {
    return 1.25 * std::pow(1.f / k, 1/(std::log2((float) n / nClusters) / std::log2(k)));
  }
  
  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(sne::MinimizationBuffers minimizationBuffers, sne::Params params, int priority)
  : RenderTask(priority, "EmbeddingRenderTask"), 
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _params(params),
    _canDrawLabels(false),
    _colorMapping(ColorMapping::labels),
    _weighForces(true),
    _weightFixed(params.k),
    _weightFalloff(calculateFalloff(params.n, params.k, params.nClusters)),
    _numClusters(params.nClusters),
    _numClustersPrev(params.nClusters),
    _pointRadius(0.003f),
    _pointOpacity(1.0f) {
    // Enable/disable render task by default
    enable = DH_VIS_EMBEDDING_INIT;

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

    // Initialize vertex array object
    {
      glCreateVertexArrays(1, &_vaoHandle);

      // Specify vertex buffers and element buffer
      constexpr uint embeddingStride = (D == 2) ? 2 : 4;
      glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(glm::vec2));               // Quad positions
      glVertexArrayVertexBuffer(_vaoHandle, 1, _minimizationBuffers.embedding, 0, embeddingStride * sizeof(float));   // Embedding positions
      glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));                                        // Quad elements/indices

      // Embedding positions advance once for the full set of (6) vertices in ePositions/quadPositions drawn
      glVertexArrayBindingDivisor(_vaoHandle, 0, 0);
      glVertexArrayBindingDivisor(_vaoHandle, 1, 1);
      
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

    _isInit = true;
  }

  template <uint D>
  EmbeddingRenderTask<D>::~EmbeddingRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
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

  template <uint D>
  void EmbeddingRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {

    if (!enable) {
      return;
    }

    glAssert();
    std::vector<vec> embedding(_params.n);
    glGetNamedBufferSubData(_minimizationBuffers.embedding, 0, _params.n * sizeof(vec), embedding.data());
    glAssert();
    std::vector<vec> bounds(4);
    glGetNamedBufferSubData(_minimizationBuffers.bounds, 0, 4 * sizeof(vec), bounds.data());
    glAssert();

    // Only allow drawing labels if a buffer is provided with said labels
    _canDrawLabels = (labelsHandle > 0);
    
    _program.bind();

    std::vector<uint> labeled(_params.n);
    glGetNamedBufferSubData(_minimizationBuffers.labeled, 0, _params.n * sizeof(uint), labeled.data());

    // Set uniforms
    _program.template uniform<float, 4, 4>("model_view", model_view);
    _program.template uniform<float, 4, 4>("proj", proj);
    _program.template uniform<float>("pointOpacity", _pointOpacity);
    _program.template uniform<float>("pointRadius", _pointRadius);
    _program.template uniform<uint>("colorMapping", _colorMapping);
    _program.template uniform<bool>("canDrawLabels", _canDrawLabels);
    _program.template uniform<bool>("selectLabeledOnly", _selectLabeledOnly);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelsHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimizationBuffers.labeled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _minimizationBuffers.selected);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _minimizationBuffers.neighborhoodPreservation);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr, _params.n);

    // Check if user changed numClusters and update weightFalloff if so
    if(_numClusters != _numClustersPrev) {
      _weightFalloff = calculateFalloff(_params.n, _params.k, _numClusters);
      _numClustersPrev = _numClusters;
    }
  }

  template <uint D>
  void EmbeddingRenderTask<D>::drawImGuiComponent() {
    if (ImGui::CollapsingHeader("Embedding render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Spacing();
      if (_canDrawLabels) {
        ImGui::Text("Color mapping:");
        if (ImGui::RadioButton("Labels", _colorMapping==ColorMapping::labels)) { _colorMapping = ColorMapping::labels; }
        if (ImGui::RadioButton("Neighborhood preservation", _colorMapping==ColorMapping::neighborhoodPreservation)) { _colorMapping = ColorMapping::neighborhoodPreservation; }
        if (ImGui::RadioButton("None", _colorMapping==ColorMapping::none)) { _colorMapping = ColorMapping::none; }
      }
      ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.0f, 1.0f);
      ImGui::SliderFloat("Point radius", &_pointRadius, 0.0001f, 0.01f, "%.4f");
      ImGui::Spacing();
      ImGui::Checkbox("Fixed datapoint force weighting", &_weighForces);
      ImGui::SliderFloat("Weight for fixed datapoints", &_weightFixed, 1.0f, _params.k * 5.0f);
      ImGui::SliderFloat("weight falloff", &_weightFalloff, 0.f, 1.f, "%.4f");
      ImGui::Text("or set the number of clusters you see:");
      ImGui::SliderInt("Number of apparent clusters", &_numClusters, 1, 50);
      ImGui::Spacing();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class EmbeddingRenderTask<2>;
  template class EmbeddingRenderTask<3>;
} // dh::vis