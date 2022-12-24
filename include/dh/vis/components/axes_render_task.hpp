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
  class AxesRenderTask : public RenderTask {
    using vec = util::AlignedVec<D, float>;

  public:
    AxesRenderTask();
    AxesRenderTask(sne::MinimizationBuffers minimization, sne::Params* params, std::vector<char> axisMapping, int selectedAxistype, int priority);
    ~AxesRenderTask();

    // Copy constr/assignment is explicitly deleted
    AxesRenderTask(const AxesRenderTask&) = delete;
    AxesRenderTask& operator=(const AxesRenderTask&) = delete;

    // Move constr/operator moves handles
    AxesRenderTask(AxesRenderTask&&) noexcept;
    AxesRenderTask& operator=(AxesRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle = 0) override;
    void drawImGuiComponent() override;
    std::vector<char> getAxisMapping() { return _axisMapping; }
    int getSelectedIndex() { return _selectedIndex; }

  private:
    // State
    bool _isInit;
    sne::MinimizationBuffers _minimization;
    sne::Params* _params;
    std::vector<char> _axisMapping;
    std::vector<char> _axisMappingPrev;

    // ImGui state
    int _selectedAxistype;
    int _selectedIndex;

    // Objects
    GLuint _vboHandlePositions;
    GLuint _vboHandleColors;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    enum ColorMapping {
      none,
      labels,
      neighborhoodPreservation
    };
    
    // std::swap impl
    friend void swap(AxesRenderTask& a, AxesRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._minimization, b._minimization);
      swap(a._params, b._params);
      swap(a._axisMapping, b._axisMapping);
      swap(a._axisMappingPrev, b._axisMappingPrev);
      swap(a._selectedAxistype, b._selectedAxistype);
      swap(a._selectedIndex, b._selectedIndex);
      swap(a._vboHandlePositions, b._vboHandlePositions);
      swap(a._vboHandleColors, b._vboHandleColors);
      swap(a._program, b._program);
      swap(a._vaoHandle, b._vaoHandle);
    }
  };
} // dh::vis