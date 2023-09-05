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

#pragma once

#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/vis/render_queue.hpp"
#include "dh/vis/components/selection_input_task.hpp"

namespace dh::vis {
  template <uint D>
  class LinkRenderTask : public RenderTask {
    using vec = util::AlignedVec<D, float>;

  public:
    LinkRenderTask();
    LinkRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, sne::SimilaritiesBuffers similaritiesBuffers);
    ~LinkRenderTask();

    // Copy constr/assignment is explicitly deleted
    LinkRenderTask(const LinkRenderTask&) = delete;
    LinkRenderTask& operator=(const LinkRenderTask&) = delete;

    // Move constr/operator moves handles
    LinkRenderTask(LinkRenderTask&&) noexcept;
    LinkRenderTask& operator=(LinkRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj) override;
    void drawImGuiComponent() override;
    void createVAO();
    void updateLinks();
    void clearLinks();
    void setSimilaritiesBuffers(sne::SimilaritiesBuffers similaritiesBuffers) { _similaritiesBuffers = similaritiesBuffers; }
    void setMinimizationBuffers(sne::MinimizationBuffers minimizationBuffers) {
      _minimizationBuffers = minimizationBuffers;
      createVAO(); // Recreate VAO with updated embedding buffer
    }

  private:
    enum class BufferType {
      eSizes,
      eScan,
      eElements,

      Length
    };

    enum class ProgramType {
      eCollectElements,
      eRender,

      Length
    };

    // State
    bool _isInit;
    sne::Params* _params;
    dh::vis::Input _input;
    uint _nLinks;
    bool _enabledPrev;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    sne::MinimizationBuffers _minimizationBuffers;
    sne::SimilaritiesBuffers _similaritiesBuffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    void setInput(dh::vis::Input input) { _input = input; }

    // std::swap impl
    friend void swap(LinkRenderTask& a, LinkRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._input, b._input);
      swap(a._nLinks, b._nLinks);
      swap(a._buffers, b._buffers);
      swap(a._minimizationBuffers, b._minimizationBuffers);
      swap(a._similaritiesBuffers, b._similaritiesBuffers);
      swap(a._programs, b._programs);
      swap(a._vaoHandle, b._vaoHandle);
    }
  };
} // dh::vis