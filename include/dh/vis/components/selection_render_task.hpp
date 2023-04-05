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
  class SelectionRenderTask : public RenderTask {

  public:
    SelectionRenderTask();
    SelectionRenderTask(sne::Params* params, int priority);
    ~SelectionRenderTask();

    // Copy constr/assignment is explicitly deleted
    SelectionRenderTask(const SelectionRenderTask&) = delete;
    SelectionRenderTask& operator=(const SelectionRenderTask&) = delete;

    // Move constr/operator moves handles
    SelectionRenderTask(SelectionRenderTask&&) noexcept;
    SelectionRenderTask& operator=(SelectionRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj) override;
    void drawImGuiComponent() override;
    void drawImGuiComponentSecondary() override;

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::Params* _params;
    glm::vec2 _mousePosScreen;
    dh::vis::Input _input;

    // ImGui state
    float _selectionRadiusRel;
    bool _selectLabeledOnly;
    std::vector<uint> _selectionCounts;
    float _similarityWeight;
    uint _buttonPressed;
    int _selectedDatapoint;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    void setSelectionRadiusRel(float selectionRadiusRel) { _selectionRadiusRel = selectionRadiusRel; }
    int getSelectionMode() { return _selectLabeledOnly; }
    void setMousePosScreen(const glm::vec2& mousePosScreen) { _mousePosScreen = mousePosScreen; }
    void setSelectionMode(bool selectLabeledOnly) { _selectLabeledOnly = selectLabeledOnly; }
    uint getButtonPressed() { return _buttonPressed; }
    float getSimilarityWeight() { return _similarityWeight; }
    void setSelectionCounts(std::vector<uint> selectionCounts) { _selectionCounts = selectionCounts; }
    void setInput(dh::vis::Input input) { _input = input; }
    int getSelectedDatapoint() { return _selectedDatapoint; }
    
    // std::swap impl
    friend void swap(SelectionRenderTask& a, SelectionRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._input, b._input);
      swap(a._selectionRadiusRel, b._selectionRadiusRel);
      swap(a._selectLabeledOnly, b._selectLabeledOnly);
      swap(a._selectionCounts, b._selectionCounts);
      swap(a._mousePosScreen, b._mousePosScreen);
      swap(a._buttonPressed, b._buttonPressed);
      swap(a._selectedDatapoint, b._selectedDatapoint);
      swap(a._similarityWeight, b._similarityWeight);
      swap(a._buffers, b._buffers);
      swap(a._vaoHandle, b._vaoHandle);
      swap(a._program, b._program);
    }
  };
} // dh::vis