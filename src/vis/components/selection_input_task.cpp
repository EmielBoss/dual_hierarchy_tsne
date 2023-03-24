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

#include <iostream>

#include <cmath>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "dh/util/aligned.hpp"
#include "dh/vis/components/selection_input_task.hpp"

namespace dh::vis {
  SelectionInputTask::SelectionInputTask()
  : InputTask(1, "SelectionInputTask"),
    _spacePressed(false),
    _rPressed(false),
    _fPressed(false),
    _ePressed(false),
    _sPressed(false),
    _dPressed(false),
    _uPressed(false),
    _xPressed(false),
    _yPressed(false),
    _zPressed(false),
    _gPressed(false),
    _pPressed(false),
    _shiftPressed(false),
    _ctrlPressed(false),
    _altPressed(false),
    _deletePressed(false),
    _insertPressed(false),
    _numPressed(-1),
    _mouseLeftPressed(false),
    _mouseRightPressed(false),
    _mouseMiddlePressed(false),
    _mouseScroll(3.0f), 
    _mouseScrollMult(0.5f),
    _mousePosMult(1.0f) {
    // ...
  }

  void SelectionInputTask::process() {
    
  }

  void SelectionInputTask::mousePosInput(double xPos, double yPos) {
    _mousePosScreen = glm::vec2(xPos, yPos);
    
    // Obtain current window handle for window size
    util::GLWindow* window = util::GLWindow::currentWindow();
    if (!window) {
      return;
    }

    // Record current position in [-1, 1]
    _mousePosClip = _mousePosScreen / glm::vec2(window->size());
    _mousePosClip = 2.0f * _mousePosClip - 1.0f;
  }

  void SelectionInputTask::mouseButtonInput(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) { _mouseLeftPressed = true; }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) { _mouseLeftPressed = false; }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) { _mouseRightPressed = true; }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) { _mouseRightPressed = false; }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) { _mouseMiddlePressed = true; }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) { _mouseMiddlePressed = false; }
  }

  void SelectionInputTask::mouseScrollInput(double xScroll, double yScroll) {
    if(!_shiftPressed) {
      _mouseScroll = std::max(0.001f, _mouseScroll + _mouseScrollMult * static_cast<float>(yScroll));
    }
  }

  void SelectionInputTask::keyboardInput(int button, int action) {
    if (button == GLFW_KEY_R && action == GLFW_PRESS) { _rPressed = true; }
    else if (button == GLFW_KEY_R && action == GLFW_RELEASE) { _rPressed = false; }

    if (button == GLFW_KEY_F && action == GLFW_PRESS) { _fPressed = true; }
    else if (button == GLFW_KEY_F && action == GLFW_RELEASE) { _fPressed = false; }

    if (button == GLFW_KEY_E && action == GLFW_PRESS) { _ePressed = true; }
    else if (button == GLFW_KEY_E && action == GLFW_RELEASE) { _ePressed = false; }

    if (button == GLFW_KEY_S && action == GLFW_PRESS) { _sPressed = true; }
    else if (button == GLFW_KEY_S && action == GLFW_RELEASE) { _sPressed = false; }

    if (button == GLFW_KEY_D && action == GLFW_PRESS) { _dPressed = true; }
    else if (button == GLFW_KEY_D && action == GLFW_RELEASE) { _dPressed = false; }

    if (button == GLFW_KEY_U && action == GLFW_PRESS) { _uPressed = true; }
    else if (button == GLFW_KEY_U && action == GLFW_RELEASE) { _uPressed = false; }

    if (button == GLFW_KEY_X && action == GLFW_PRESS) { _xPressed = true; }
    else if (button == GLFW_KEY_X && action == GLFW_RELEASE) { _xPressed = false; }

    if (button == GLFW_KEY_Y && action == GLFW_PRESS) { _yPressed = true; }
    else if (button == GLFW_KEY_Y && action == GLFW_RELEASE) { _yPressed = false; }

    if (button == GLFW_KEY_Z && action == GLFW_PRESS) { _zPressed = true; }
    else if (button == GLFW_KEY_Z && action == GLFW_RELEASE) { _zPressed = false; }

    if (button == GLFW_KEY_G && action == GLFW_PRESS) { _gPressed = true; }
    else if (button == GLFW_KEY_G && action == GLFW_RELEASE) { _gPressed = false; }

    if (button == GLFW_KEY_P && action == GLFW_PRESS) { _pPressed = true; }
    else if (button == GLFW_KEY_P && action == GLFW_RELEASE) { _pPressed = false; }

    if (button == GLFW_KEY_SPACE && action == GLFW_PRESS) { _spacePressed = true; }
    else if (button == GLFW_KEY_SPACE && action == GLFW_RELEASE) { _spacePressed = false; }

    if (button == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) { _ctrlPressed = true; }
    else if (button == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) { _ctrlPressed = false; }

    if (button == GLFW_KEY_LEFT_SHIFT && action == GLFW_PRESS) { _shiftPressed = true; }
    else if (button == GLFW_KEY_LEFT_SHIFT && action == GLFW_RELEASE) { _shiftPressed = false; }

    if (button == GLFW_KEY_LEFT_ALT && action == GLFW_PRESS) { _altPressed = true; }
    else if (button == GLFW_KEY_LEFT_ALT && action == GLFW_RELEASE) { _altPressed = false; }

    if (button == GLFW_KEY_DELETE && action == GLFW_PRESS) { _deletePressed = true; }
    else if (button == GLFW_KEY_DELETE && action == GLFW_RELEASE) { _deletePressed = false; }

    if (button == GLFW_KEY_INSERT && action == GLFW_PRESS) { _insertPressed = true; }
    else if (button == GLFW_KEY_INSERT && action == GLFW_RELEASE) { _insertPressed = false; }

    if (button == GLFW_KEY_0 && action == GLFW_PRESS) { _numPressed = 0; }
    else if (button == GLFW_KEY_1 && action == GLFW_PRESS) { _numPressed = 1; }
    else if (button == GLFW_KEY_2 && action == GLFW_PRESS) { _numPressed = 2; }
    else if (button == GLFW_KEY_3 && action == GLFW_PRESS) { _numPressed = 3; }
    else if (button == GLFW_KEY_4 && action == GLFW_PRESS) { _numPressed = 4; }
    else if (button == GLFW_KEY_5 && action == GLFW_PRESS) { _numPressed = 5; }
    else if (button == GLFW_KEY_6 && action == GLFW_PRESS) { _numPressed = 6; }
    else if (button == GLFW_KEY_7 && action == GLFW_PRESS) { _numPressed = 7; }
    else if (button == GLFW_KEY_8 && action == GLFW_PRESS) { _numPressed = 8; }
    else if (button == GLFW_KEY_9 && action == GLFW_PRESS) { _numPressed = 9; }
    else if ((button == GLFW_KEY_0 || button == GLFW_KEY_1 || button == GLFW_KEY_2 || button == GLFW_KEY_3 || button == GLFW_KEY_4 || button == GLFW_KEY_5 || button == GLFW_KEY_6 || button == GLFW_KEY_7 || button == GLFW_KEY_8 || button == GLFW_KEY_9)
                                  && action == GLFW_RELEASE) { _numPressed = -1; }
  }

} // dh::vis