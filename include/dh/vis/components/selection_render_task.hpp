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
    SelectionRenderTask(std::array<GLuint, 7> texturedataBuffers, GLuint attributeWeights, sne::Params* params, int priority, const float* _dataPtr);
    ~SelectionRenderTask();

    // Copy constr/assignment is explicitly deleted
    SelectionRenderTask(const SelectionRenderTask&) = delete;
    SelectionRenderTask& operator=(const SelectionRenderTask&) = delete;

    // Move constr/operator moves handles
    SelectionRenderTask(SelectionRenderTask&&) noexcept;
    SelectionRenderTask& operator=(SelectionRenderTask&&) noexcept;

    void averageSelectedImages();
    void clearSelection(bool imageDataset);
    void render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle = 0) override;
    void drawImGuiComponent() override;
    void drawImGuiComponentSecondary() override;

    void drawImGuiTab(uint selectionIndex, uint typeIndex, const char* text);
    void drawImGuiTexture(GLuint textureHandle);
    void drawImGuiTextureControls();
    void drawImPlotBarPlot(uint selectionIndex);
    void setSelectionCounts(std::vector<uint> selectionCounts) { _selectionCounts = selectionCounts; }
    void setInput(dh::vis::Input input) { _input = input; }

    float getBufferValue(GLuint buffer, int index);
    bool getHoveringTexture() { return _hoveringTexture; }
    int getSelectedDatapoint() { return _selectedDatapoint; }

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    enum class SelectionAttributesType {
      eAveragePrimary,
      eVariancePrimary,
      eAverageSecondary,
      eVarianceSecondary,
      eAverageDifference,
      eVarianceDifference,
      eOverlay,

      Length
    };

    // State
    bool _isInit;
    sne::Params* _params;
    glm::vec2 _mousePosScreen;
    dh::vis::Input _input;

    // ImGui state
    bool _canDrawLabels;
    float _selectionRadiusRel;
    bool _selectLabeledOnly;
    std::vector<uint> _selectionCounts;
    int _draggedTexel;
    float _similarityWeight;
    float _attributeWeight;
    int _texelBrushRadius;
    float _autoselectPercentage;
    uint _buttonPressed;
    int _selectedDatapoint;
    bool _selectAll;
    bool _hoveringTexture;
    uint _textureTabOpened;
    bool _plotError;
    uint _currentSelectionTab;
    uint _currentTypeTab;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<SelectionAttributesType, GLuint> _textures;
    std::array<GLuint, 7> _texturedataBuffers;
    GLuint _attributeWeightsBuffer;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    void setSelectionRadiusRel(float selectionRadiusRel) { _selectionRadiusRel = selectionRadiusRel; }
    int getSelectionMode() { return _selectLabeledOnly; }
    void setMousePosScreen(const glm::vec2& mousePosScreen) { _mousePosScreen = mousePosScreen; }
    void setSelectionMode(bool selectLabeledOnly) { _selectLabeledOnly = selectLabeledOnly; }
    int getDraggedTexel() { return _draggedTexel; }
    uint getButtonPressed() { return _buttonPressed; }
    bool getSelectAll() { return _selectAll; }
    float getAttributeWeight() { return _attributeWeight; }
    int getTexelBrushRadius() { return _texelBrushRadius; }
    float getSimilarityWeight() { return _similarityWeight; }
    float getAutoselectPercentage() { return _autoselectPercentage; }
    float getOpenedTextureIndex() { return _currentSelectionTab * 2 + _currentTypeTab; }
    
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
      swap(a._draggedTexel, b._draggedTexel);
      swap(a._buttonPressed, b._buttonPressed);
      swap(a._selectAll, b._selectAll);
      swap(a._selectedDatapoint, b._selectedDatapoint);
      swap(a._attributeWeight, b._attributeWeight);
      swap(a._texelBrushRadius, b._texelBrushRadius);
      swap(a._similarityWeight, b._similarityWeight);
      swap(a._autoselectPercentage, b._autoselectPercentage);
      swap(a._plotError, b._plotError);
      swap(a._buffers, b._buffers);
      swap(a._texturedataBuffers, b._texturedataBuffers);
      swap(a._textures, b._textures);
      swap(a._attributeWeightsBuffer, b._attributeWeightsBuffer);
      swap(a._vaoHandle, b._vaoHandle);
      swap(a._program, b._program);
    }
  };
} // dh::vis