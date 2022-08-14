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
  class SelectionRenderTask : public RenderTask {

  public:
    SelectionRenderTask();
    SelectionRenderTask(std::array<GLuint, 2> textures, std::array<GLuint, 2> texturedataBuffers, GLuint attributeWeights, sne::Params params, int priority, const float* _dataPtr);
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
    void drawImGuiImageButton(GLuint textureHandle);
    void setSelectionCount(uint selectionCount) { _selectionCount = selectionCount; }
    float getBufferValue(GLuint buffer, int index);
    bool getHoveringTexture() { return _hoveringTexture; }
    bool getHoveringTexturePrev() { return _hoveringTexturePrev; }

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::Params _params;
    glm::vec2 _mousePosScreen;

    // ImGui state
    bool _canDrawLabels;
    float _selectionRadiusRel;
    bool _selectLabeledOnly;
    uint _selectionCount;
    int _draggedAttribute;
    float _attributeWeight;
    uint _buttonPressed;
    bool _selectAll;
    bool _hoveringTexture;
    bool _hoveringTexturePrev;
    float _perplexity;
    int _k;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    std::array<GLuint, 2> _texturedataBuffers;
    std::array<GLuint, 2> _textures;
    GLuint _attributeWeightsBuffer;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    float getSelectionRadiusRel() { return _selectionRadiusRel; }
    void setSelectionRadiusRel(float selectionRadiusRel) { _selectionRadiusRel = selectionRadiusRel; }
    int getSelectionMode() { return _selectLabeledOnly; }
    void setMousePosScreen(const glm::vec2& mousePosScreen) { _mousePosScreen = mousePosScreen; }
    void setSelectionMode(bool selectLabeledOnly) { _selectLabeledOnly = selectLabeledOnly; }
    int getDraggedAttribute() { return _draggedAttribute; }
    uint getButtonPressed() { return _buttonPressed; }
    bool getSelectAll() { return _selectAll; }
    float getAttributeWeight() { return _attributeWeight; }
    float getPerplexity() { return _perplexity; }
    uint getK() { return (uint) _k; }
    
    // std::swap impl
    friend void swap(SelectionRenderTask& a, SelectionRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._selectionRadiusRel, b._selectionRadiusRel);
      swap(a._selectLabeledOnly, b._selectLabeledOnly);
      swap(a._selectionCount, b._selectionCount);
      swap(a._mousePosScreen, b._mousePosScreen);
      swap(a._draggedAttribute, b._draggedAttribute);
      swap(a._buttonPressed, b._buttonPressed);
      swap(a._attributeWeight, b._attributeWeight);
      swap(a._perplexity, b._perplexity);
      swap(a._k, b._k);
      swap(a._buffers, b._buffers);
      swap(a._texturedataBuffers, b._texturedataBuffers);
      swap(a._textures, b._textures);
      swap(a._attributeWeightsBuffer, b._attributeWeightsBuffer);
      swap(a._vaoHandle, b._vaoHandle);
      swap(a._program, b._program);
    }
  };
} // dh::vis