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

namespace dh::vis {
  template <uint D>
  class EmbeddingHierarchyRenderTask : public RenderTask {
    using vec = util::AlignedVec<D, float>;

  public:
    EmbeddingHierarchyRenderTask();
    EmbeddingHierarchyRenderTask(sne::MinimizationBuffers minimization, sne::EmbeddingHierarchyBuffers embeddingHierarchy, sne::Params* params, int priority);
    ~EmbeddingHierarchyRenderTask();

    // Copy constr/assignment is explicitly deleted
    EmbeddingHierarchyRenderTask(const EmbeddingHierarchyRenderTask&) = delete;
    EmbeddingHierarchyRenderTask& operator=(const EmbeddingHierarchyRenderTask&) = delete;

    // Move constr/operator moves handles
    EmbeddingHierarchyRenderTask(EmbeddingHierarchyRenderTask&&) noexcept;
    EmbeddingHierarchyRenderTask& operator=(EmbeddingHierarchyRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj) override;
    void drawImGuiComponent() override;

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::MinimizationBuffers _minimization;
    sne::EmbeddingHierarchyBuffers _embeddingHierarchy;
    sne::Params* _params;

    // ImGui state
    float _lineWidth;
    glm::vec4 _lineColor;
    bool _selectLvl;
    uint _selectedLvl;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }
    
    // std::swap impl
    friend void swap(EmbeddingHierarchyRenderTask& a, EmbeddingHierarchyRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._minimization, b._minimization);
      swap(a._embeddingHierarchy, b._embeddingHierarchy);
      swap(a._params, b._params);
      swap(a._lineWidth, b._lineWidth);
      swap(a._lineColor, b._lineColor);
      swap(a._selectLvl, b._selectLvl);
      swap(a._selectedLvl, b._selectedLvl);
      swap(a._buffers, b._buffers);
      swap(a._program, b._program);
      swap(a._vaoHandle, b._vaoHandle);
    }
  };
} // dh::vis