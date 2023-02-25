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
#include "dh/vis/components/trackball_input_task.hpp"

namespace dh::vis {
  TrackballInputTask::TrackballInputTask()
  : InputTask(2, "TrackballInputTask"),
    _mouseTrackState(false), 
    _mouseScrollState(3.0f), 
    _mousePosState(0.0f), 
    _mousePosStatePrev(0.0f),
    _lookatState(1),
    _matrix(1),
    _mouseScrollMult(0.5f),
    _mousePosMult(1.0f),
    _shiftPressed(false),
    _cameraPos(0, 0, 3, 1),
    _up(0, 1, 0, 1) {
    // ...
  }

  void TrackballInputTask::process() {
    float time = glfwGetTime();
    float timeDelta = time - _timePrev;
    _timePrev = time;

    // Compute extremely simplified plane rotation
    if (_mouseTrackState && _mousePosState != _mousePosStatePrev) {
      glm::vec2 v = _mousePosMult * (_mousePosState - _mousePosStatePrev) * timeDelta * 123.f; // Multiplying by timeDelta * 123 keeps movement consistent when performance inevitably turns to shit
      glm::mat4 rx = glm::rotate(v.x, glm::vec3(0, 1, 0));
      glm::mat4 ry = glm::rotate(v.y, glm::vec3(1, 0, 0));
      _lookatState = ry * rx * _lookatState;
      _cameraPos = glm::inverse(ry * rx) * _cameraPos;
      _up = glm::inverse(ry * rx) * _up;
    }

    // Apply mouse scroll for final result
    _matrix = glm::lookAt(
      glm::vec3(0, 0, 1) * _mouseScrollState, // Camera position
      glm::vec3(0, 0, 0),                     // Target position
      glm::vec3(0, 1, 0)                      // World up
    ) * _lookatState;

    // Create change of base matrix from world to camera space (only used for translation)
    glm::vec3 direction = -glm::normalize(glm::vec3(_cameraPos));
    glm::vec3 right = glm::normalize(glm::cross(direction, glm::vec3(_up)));
    glm::vec3 up = glm::cross(right, direction);
    _changeOfBaseToCam = glm::transpose(glm::mat3(right, up, direction));
  }

  void TrackballInputTask::mousePosInput(double xPos, double yPos) {
    // Obtain current window handle for window size
    util::GLWindow* window = util::GLWindow::currentWindow();
    if (!window) {
      return;
    }

    // Record previous position as last recorded position
    _mousePosStatePrev = _mousePosState;

    // Recorcd current position in [-1, 1]
    _mousePosState = glm::vec2(xPos, yPos)
                   / glm::vec2(window->size());
    _mousePosState = 2.0f * _mousePosState - 1.0f;
  }

  void TrackballInputTask::mouseButtonInput(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) { _mouseTrackState = true; } else
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) { _mouseTrackState = false; }
  }

  void TrackballInputTask::mouseScrollInput(double xScroll, double yScroll) {
    if(_shiftPressed) {
    _mouseScrollState = std::max(0.001f, _mouseScrollState - _mouseScrollMult * static_cast<float>(yScroll));
    }
  }

  void TrackballInputTask::keyboardInput(int button, int action) {
    if (button == GLFW_KEY_LEFT_SHIFT && action == GLFW_PRESS) { _shiftPressed = true; }
    else if (button == GLFW_KEY_LEFT_SHIFT && action == GLFW_RELEASE) { _shiftPressed = false; }

    if (button == GLFW_KEY_HOME && action == GLFW_PRESS) {
      _mouseScrollState = 3.0f; 
      _lookatState = glm::mat4(1);
    }

    if (button == GLFW_KEY_END && action == GLFW_PRESS) {
      _mouseScrollState = 3.0f; 
      _lookatState = glm::rotate(glm::radians(180.f), glm::vec3(0, 1, 0));
    }

    if (button == GLFW_KEY_KP_4 && action == GLFW_PRESS) {
      glm::vec4 axis = _lookatState * glm::vec4(0, 1, 0, 1);
      _lookatState = glm::rotate(glm::radians(90.f), glm::vec3(axis)) * _lookatState;
    }

    if (button == GLFW_KEY_KP_6 && action == GLFW_PRESS) {
      glm::vec4 axis = _lookatState * glm::vec4(0, 1, 0, 1);
      _lookatState = glm::rotate(glm::radians(-90.f), glm::vec3(axis)) * _lookatState;
    }

    if (button == GLFW_KEY_KP_2 && action == GLFW_PRESS) {
      glm::vec4 axis = _lookatState * glm::vec4(1, 0, 0, 1);
      _lookatState = glm::rotate(glm::radians(-90.f), glm::vec3(axis)) * _lookatState;
    }

    if (button == GLFW_KEY_KP_8 && action == GLFW_PRESS) {
      glm::vec4 axis = _lookatState * glm::vec4(1, 0, 0, 1);
      _lookatState = glm::rotate(glm::radians(90.f), glm::vec3(axis)) * _lookatState;
    }
  }

} // dh::vis