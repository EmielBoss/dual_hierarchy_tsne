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

  float calculateFalloff(uint n, uint k, int nClusters) {
    return 1.25 * std::pow(1.f / k, 1/(std::log2((float) n / nClusters) / std::log2(k)));
  }
  
  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(sne::MinimizationBuffers minimizationBuffers, sne::Params* params,
                                              std::vector<GLuint> classTextures, std::vector<uint> classCounts, int priority)
  : RenderTask(priority, "EmbeddingRenderTask"), 
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _classTextures(classTextures),
    _classCounts(classCounts),
    _classIsSet(_params->nClasses, false),
    _classesSet(),
    _setChanged(false),
    _nSelectedNeighbors(0),
    _params(params),
    _colorMapping(ColorMapping::labels),
    _weighForces(true),
    _weightFixed(params->k),
    _weightFalloff(calculateFalloff(params->n, params->k, params->nClusters)),
    // _numClusters(params->nClusters),
    // _numClustersPrev(params->nClusters),
    _pointRadius(100.f / _params->n),
    _pointOpacity(1.0f),
    _buttonPressed(0),
    _classButtonPressed(-1),
    _perplexity(params->perplexity),
    _k((int) _params->k),
    _iteration(0),
    _klDivergence(-1.f) {
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

    createVAO();

    _classNames = std::vector<std::string>(_params->nClasses, "");
    dh::util::readTxtClassNames(_params->datasetName + ".txt", _classNames, _params->nClasses);

    generateClassColors();
    _isInit = true;
  }

  template <uint D>
  EmbeddingRenderTask<D>::~EmbeddingRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteBuffers(_classTextures.size(), _classTextures.data());
      glDeleteBuffers(1, &_colorBuffer);
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

  template <uint D>
  void EmbeddingRenderTask<D>::generateClassColors() {
    
    // _colors = {
    //   glm::vec4(16, 78, 139, 255),
    //   glm::vec4(0, 128, 0, 255),
    //   glm::vec4(139, 90, 43, 255),
    //   glm::vec4(138, 43, 226, 255),
    //   glm::vec4(255, 150, 0, 255),
    //   glm::vec4(204, 40, 40, 255),
    //   glm::vec4(131, 139, 131, 255),
    //   glm::vec4(0, 205, 0, 255),
    //   glm::vec4(0, 150, 255, 255),
    //   glm::vec4(220, 220, 220, 255)
    // };

    _colors = {
      glm::vec4(230, 25, 75, 255),
      glm::vec4(60, 180, 75, 255),
      glm::vec4(255, 225, 25, 255),
      glm::vec4(0, 130, 200, 255),
      glm::vec4(245, 130, 48, 255),
      glm::vec4(0, 128, 128, 255),
      glm::vec4(70, 240, 240, 255),
      glm::vec4(145, 30, 180, 255),
      glm::vec4(250, 190, 212, 255),
      glm::vec4(170, 110, 40, 255),
      glm::vec4(255, 250, 200, 255),
      glm::vec4(128, 0, 0, 255),
      glm::vec4(170, 255, 195, 255),
      glm::vec4(128, 128, 0, 255),
      glm::vec4(255, 215, 180, 255),
      glm::vec4(0, 0, 128, 255),
      glm::vec4(128, 128, 128, 255)
    };

    int nColorsToAdd = _params->nClasses - _colors.size();
    for(int i = 0; i < nColorsToAdd; ++i) {
      glm::vec4 newColor = _colors[i] + _colors[i+1];
      newColor /= 2.f;
      _colors.push_back(newColor);
    }

    glCreateBuffers(1, &_colorBuffer);
    glNamedBufferStorage(_colorBuffer, _params->nClasses * sizeof(glm::vec4), _colors.data(), GL_DYNAMIC_STORAGE_BIT);
    glAssert();
  }

  template <uint D>
  void EmbeddingRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj) {

    if (!enable) {
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
    _program.template uniform<float>("divisor", divisor);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.labels);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimizationBuffers.labeled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _minimizationBuffers.disabled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _minimizationBuffers.fixed);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _minimizationBuffers.selection);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _minimizationBuffers.neighborhoodPreservation);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _colorBuffer);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr, _params->n);

    // Check if user changed numClusters and update weightFalloff if so
    if(_numClusters != _numClustersPrev) {
      _weightFalloff = calculateFalloff(_params->n, _params->k, _numClusters);
      _numClustersPrev = _numClusters;
    }
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
        ImGui::Text("Color mapping:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Labels", _colorMapping==ColorMapping::labels)) { _colorMapping = ColorMapping::labels; }
        ImGui::SameLine();
        if (ImGui::RadioButton("Neighborhood preser.", _colorMapping==ColorMapping::neighborhoodPreservation)) { _colorMapping = ColorMapping::neighborhoodPreservation; }
        ImGui::SameLine();
        if (ImGui::RadioButton("None", _colorMapping==ColorMapping::none)) { _colorMapping = ColorMapping::none; }
      }
      ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.0f, 1.0f);
      ImGui::SliderFloat("Point radius", &_pointRadius, 0.0001f, 0.01f, "%.4f");
      ImGui::Spacing();
      ImGui::Checkbox("Fixed datapoint force weighting", &_weighForces);
      ImGui::SliderFloat("Fixed datapoint weight", &_weightFixed, 1.0f, _params->k * 5.0f);
      ImGui::SliderFloat("weight falloff", &_weightFalloff, 0.f, 1.f, "%.4f");
      // ImGui::Text("or set the number of clusters you see:");
      // ImGui::SliderInt("Number of apparent clusters", &_numClusters, 1, 50);
      ImGui::Spacing();

      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.25f);
      ImGui::SliderFloat("Perpl.", &_perplexity, 1.0f, 100.f);
      if(ImGui::IsItemHovered() && ImGui::IsItemActive()) { _k = (int) std::min(_params->kMax, 3 * (uint)(_perplexity) + 1); }
      ImGui::SameLine(); ImGui::SliderInt("k", &_k, 2, _params->kMax);
      if(ImGui::SameLine(); ImGui::Button("Focus")) { _buttonPressed = 1; }
      if(ImGui::IsItemHovered()) { ImGui::BeginTooltip(); ImGui::Text("Restarts minimization with only the selected datapoints and hyperparameters."); ImGui::EndTooltip(); }
      ImGui::PopItemWidth();
    }
  }

  // Draws the class list
  template <uint D>
  void EmbeddingRenderTask<D>::drawImGuiComponentSecondary() {
    if(_minimizationBuffers.labels == 0) { return; }
    _classButtonPressed = -1;
    _setChanged = false;
    if (ImGui::CollapsingHeader("Classes", ImGuiTreeNodeFlags_DefaultOpen)) {
      for(uint i = 0; i < _params->nClasses; ++i) {
        if(ImGui::ImageButton((void*)(intptr_t)_classTextures[i], ImVec2(19, 19), ImVec2(0,0), ImVec2(1,1), 0)) { _classButtonPressed = i; }
        ImVec4 color = ImVec4(_colors[i].x / 400.f, _colors[i].y / 400.f, _colors[i].z / 400.f, _colors[i].w / 255.f);
        std::string leadingZeros = i < 10 ? "0" : "";
        std::string text = leadingZeros + std::to_string(i) + " | " + std::to_string(_classCounts[i]) + " " + _classNames[i];
        if(ImGui::SameLine(); ImGui::ColorEdit3(text.c_str(), (float*) &color, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder)) {
          glm::vec4 colorUpdated = glm::vec4(color.x * 400.f, color.y * 400.f, color.z * 400.f, 255.f);
          _colors[i] = colorUpdated;
          glNamedBufferSubData(_colorBuffer, i * sizeof(glm::vec4), sizeof(glm::vec4), &colorUpdated);
        }
        ImGui::SameLine();
        if(!_classIsSet[i]) {
          std::string label = "Set##" + std::to_string(i);
          if(ImGui::Button(label.c_str())) { setClass(i); }
        } else {
          std::string label = "Unset##" + std::to_string(i);
          if(ImGui::Button(label.c_str())) { unsetClass(i); }
        }
      }

      if(_nSelectedNeighbors > 0) {
        ImGui::Text("No. of selected neighbours: %u", _nSelectedNeighbors);
        ImGui::SameLine();
        if(_classesSet.size() == 2) { ImGui::Text("(interclass)"); } else { ImGui::Text("(all)"); }
      } else { ImGui::Dummy(ImVec2(43.0f, 13.0f)); }
    }
  }

  template <uint D>
  void EmbeddingRenderTask<D>::setClass(int c) {
    if(_classesSet.size() == 2) {
      int classPopped = *_classesSet.begin();
      _classesSet.erase(_classesSet.begin());
      _classIsSet[classPopped] = false;
    }
    _classIsSet[c] = true;
    _classesSet.insert(c);
    _setChanged = true;
  }

  template <uint D>
  void EmbeddingRenderTask<D>::unsetClass(int c) {
    _classIsSet[c] = false;
    _classesSet.erase(c);
    _setChanged = true;
  }

  // Template instantiations for 2/3 dimensions
  template class EmbeddingRenderTask<2>;
  template class EmbeddingRenderTask<3>;
} // dh::vis