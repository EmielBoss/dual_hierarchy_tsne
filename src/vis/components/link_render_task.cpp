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
    constexpr uint embeddingStride = (D == 2) ? 2 : 4;
    glVertexArrayVertexBuffer(_vaoHandle, 0, _minimizationBuffers.embeddingRel, 0, embeddingStride * sizeof(float));    // Embedding positions
    glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));                                             // Elements/indices
    
    // Specify vertex array data organization
    constexpr uint embeddingSize = (D == 2) ? 2 : 3;
    glVertexArrayAttribFormat(_vaoHandle, 0, D, GL_FLOAT, GL_FALSE, 0);

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

    glDeleteBuffers(1, &_buffers(BufferType::eElements));
    glCreateBuffers(1, &_buffers(BufferType::eElements));
    glNamedBufferStorage(_buffers(BufferType::eElements), _nLinks * 2 * sizeof(uint), nullptr, 0);
    { 
      auto& program = _programs(ProgramType::eCollectElements);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimizationBuffers.selection);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimizationBuffers.disabled);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similaritiesBuffers.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eElements));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1); // One warp/subgroup per datapoint
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert();
    }

    glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));
    
    glAssert();
  }

  template <uint D>
  void LinkRenderTask<D>::clearLinks() {
    _nLinks = 0;
  }

  template <uint D>
  void LinkRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj) {
    if (enabled && !_enabledPrev) { updateLinks(); }
    _enabledPrev = enabled;
    if (!enabled) { return; }
    
    _programs(ProgramType::eRender).bind();

    // Set uniforms
    _programs(ProgramType::eRender).template uniform<float, 4, 4>("model_view", model_view);
    _programs(ProgramType::eRender).template uniform<float, 4, 4>("proj", proj);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElements(GL_LINES, _nLinks * 2, GL_UNSIGNED_INT, nullptr);

    glAssert();
  }

  template <uint D>
  void LinkRenderTask<D>::drawImGuiComponent() {
    
  }


  // Template instantiations for 2/3 dimensions
  template class LinkRenderTask<2>;
  template class LinkRenderTask<3>;
} // dh::vis