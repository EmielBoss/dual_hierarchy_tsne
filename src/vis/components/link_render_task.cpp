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
#include <algorithm>
#include <numeric>
#include <string>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include <imgui.h>
#include "dh/util/gl/buffertools.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/cu/scan.cuh"
#include "dh/vis/components/link_render_task.hpp"
#include "dh/util/io.hpp" //

namespace dh::vis {
  
  template <uint D>
  LinkRenderTask<D>::LinkRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  LinkRenderTask<D>::LinkRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, sne::SimilaritiesBuffers similaritiesBuffers)
  : RenderTask(priority, "LinkRenderTask"),
    _enabledPrev(false),
    _isInit(false),
    _minimizationBuffers(minimizationBuffers),
    _similaritiesBuffers(similaritiesBuffers),
    _params(params),
    _linkOpacity(1.f),
    _interOnly(true),
    _vizSimilarities(true),
    _colorMapping(1),
    _nLinks(6) {

    // Initialize shader program
    {
      if constexpr (D == 2) {
        _programs(ProgramType::eRender).addShader(util::GLShaderType::eVertex, rsrc::get("vis/link/link.vert"));
        _programs(ProgramType::eRender).addShader(util::GLShaderType::eFragment, rsrc::get("vis/link/link.frag"));
      } else if constexpr (D == 3) {
        // _programs(ProgramType::eRender).addShader(util::GLShaderType::eVertex, rsrc::get("vis/link/3D/link.vert"));
        // _programs(ProgramType::eRender).addShader(util::GLShaderType::eFragment, rsrc::get("vis/link/3D/link.frag"));
      }
      _programs(ProgramType::eCollectElements).addShader(util::GLShaderType::eCompute, rsrc::get("vis/link/collect_elements.comp"));
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eSizes), _params->n * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eScan), _params->n * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSimilarityMin), _params->n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSimilarityMax), _params->n * sizeof(float), nullptr, 0);
      glAssert();
    }

    createVAO();

    _isInit = true;
    enabled = false;
  }

  template <uint D>
  LinkRenderTask<D>::~LinkRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
    }
  }

  template <uint D>
  LinkRenderTask<D>::LinkRenderTask(LinkRenderTask<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  LinkRenderTask<D>& LinkRenderTask<D>::operator=(LinkRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // (Re)create and initialize VAO
  template <uint D>
  void LinkRenderTask<D>::createVAO() {
    glDeleteVertexArrays(1, &_vaoHandle);
    glCreateVertexArrays(1, &_vaoHandle);

    // Specify vertex buffers and element buffer
    // glVertexArrayVertexBuffer(_vaoHandle, 0, _buffersCyclical(BufferCyclicalType::eElements), 0, sizeof(float));
    
    // Specify vertex array data organization
    glVertexArrayAttribFormat(_vaoHandle, 0, 1, GL_FLOAT, GL_FALSE, 0);

    // Other VAO properties
    glEnableVertexArrayAttrib(_vaoHandle, 0);
    glVertexArrayAttribBinding(_vaoHandle, 0, 0);
    
    glAssert();
  }

  template <uint D>
  void LinkRenderTask<D>::updateLinks() {
    if (!enabled) { return; }

    dh::util::BufferTools::instance().subsample(_similaritiesBuffers.layout, _params->n, 2, 2, _buffers(BufferType::eSizes));
    dh::util::BufferTools::instance().set(_buffers(BufferType::eSizes), _params->n, 0, 0, _minimizationBuffers.selection);
    util::Scan(_buffers(BufferType::eSizes), _buffers(BufferType::eScan), _params->n).comp(false);
    glGetNamedBufferSubData(_buffers(BufferType::eScan), (_params->n - 1) * sizeof(uint), sizeof(uint), &_nLinks);
    if(_nLinks == 0) { return; }

    dh::util::BufferTools::instance().reducePerDatapoint<float>(_similaritiesBuffers.similarities, 1, _params->n, _buffers(BufferType::eSimilarityMin), _similaritiesBuffers.layout, _similaritiesBuffers.neighbors);
    dh::util::BufferTools::instance().reducePerDatapoint<float>(_similaritiesBuffers.similarities, 2, _params->n, _buffers(BufferType::eSimilarityMax), _similaritiesBuffers.layout, _similaritiesBuffers.neighbors);

    glDeleteBuffers(_buffersCyclical.size(), _buffersCyclical.data());
    glCreateBuffers(_buffersCyclical.size(), _buffersCyclical.data());
    glNamedBufferStorage(_buffersCyclical(BufferCyclicalType::eElements), _nLinks * 2 * sizeof(uint), nullptr, 0);
    glNamedBufferStorage(_buffersCyclical(BufferCyclicalType::eSimilaritiesRel), _nLinks * sizeof(float), nullptr, 0);

    { 
      auto& program = _programs(ProgramType::eCollectElements);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<bool>("inter", _interOnly && _secondarySelectionCount > 0);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.disabled);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similaritiesBuffers.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.similarities);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eSimilarityMin));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eSimilarityMax));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffersCyclical(BufferCyclicalType::eElements));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffersCyclical(BufferCyclicalType::eSimilaritiesRel));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1); // One warp/subgroup per datapoint
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    glVertexArrayVertexBuffer(_vaoHandle, 0, _buffersCyclical(BufferCyclicalType::eElements), 0, sizeof(float));

    glAssert();
  }

  template <uint D>
  void LinkRenderTask<D>::clearLinks() {
    _nLinks = 0;
  }

  // Rendering links is done in a very hacky way: rather than the datapoint positions,
  // elements are bound as a vertex object array, and then these elements are used to
  // index into the vec2 positions and float link opacities
  template <uint D>
  void LinkRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj) {
    if (enabled && !_enabledPrev || _interOnly != _interOnlyPrev) { updateLinks(); }
    _enabledPrev = enabled;
    _interOnlyPrev = _interOnly;
    if (!enabled) { return; }
    
    _programs(ProgramType::eRender).bind();

    // Set uniforms
    _programs(ProgramType::eRender).template uniform<float, 4, 4>("model_view", model_view);
    _programs(ProgramType::eRender).template uniform<float, 4, 4>("proj", proj);
    _programs(ProgramType::eRender).template uniform<float>("linkOpacity", _linkOpacity);
    _programs(ProgramType::eRender).template uniform<uint>("colorMapping", _colorMapping);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.embeddingRel);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersCyclical(BufferCyclicalType::eSimilaritiesRel));

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawArrays(GL_LINES, 0, _nLinks * 2);

    glAssert();
  }

  template <uint D>
  void LinkRenderTask<D>::drawImGuiComponent() {
    ImGui::SliderFloat("Line opacity", &_linkOpacity, 0.0f, 2.0f);
    ImGui::Checkbox("Interlinks only", &_interOnly);
    ImGui::Checkbox("Visualize similarity", &_vizSimilarities);
    if(_vizSimilarities) {
      ImGui::SameLine();
      ImGui::Text("as");
      ImGui::SameLine();
      if (ImGui::RadioButton("red/blue", _colorMapping==ColorMapping::colors)) { _colorMapping = ColorMapping::colors; }
      ImGui::SameLine();
      if (ImGui::RadioButton("opacity", _colorMapping==ColorMapping::opacity)) { _colorMapping = ColorMapping::opacity; }
      ImGui::SameLine();
      if (ImGui::RadioButton("both", _colorMapping==ColorMapping::both)) { _colorMapping = ColorMapping::both; }
    }
    else {
      _colorMapping = ColorMapping::none;
    }
  }


  // Template instantiations for 2/3 dimensions
  template class LinkRenderTask<2>;
  template class LinkRenderTask<3>;
} // dh::vis