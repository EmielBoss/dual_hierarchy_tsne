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

#include "dh/vis/input_queue.hpp"
#include "dh/util/aligned.hpp"

namespace dh::vis {
  struct Input {
    glm::vec2 mousePos;
    float mouseScroll;
    bool mouseLeft;
    bool mouseRight;
    bool mouseMiddle;
    bool r;
    bool d;
    bool space;
    int num;
  };

  class SelectionInputTask : public InputTask {
  
  public:
    SelectionInputTask();

    void process() override;
    void mousePosInput(double xPos, double yPos) override;
    void mouseButtonInput(int button, int action) override;
    void keyboardInput(int button, int action) override;
    void mouseScrollInput(double xScroll, double yScroll) override;

    // Getters
    Input getInput() const {
      return {
        _mousePos,
        _mouseScroll,
        _mouseLeftPressed,
        _mouseRightPressed,
        _mouseMiddlePressed,
        _rPressed,
        _dPressed,
        _spacePressed,
        _numPressed
      };
    }

    glm::vec2 getMousePos() const { return _mousePos; }
    glm::vec2 getMousePosPixel() const { return _mousePosPixel; }
    bool getSpacePressed() const { return _spacePressed; }
    bool getDPressed() const { return _dPressed; }
    bool getMousePressed() const { return _mouseLeftPressed; }
    int getNumPressed() const { return _numPressed; }
    void setNumPressed(int num) { _numPressed = num; }
    void setMouseScroll(float mouseScroll) { _mouseScroll = mouseScroll; }

  private:
    // State
    bool _spacePressed; // Whether spacebar is pressed
    bool _rPressed;
    bool _dPressed; // Whether d is pressed
    bool _mouseLeftPressed; // Whether the left mouse is clicked
    bool _mouseRightPressed;
    bool _mouseMiddlePressed;
    int _numPressed;
    float _mouseScroll;
    glm::vec2 _mousePosPixel; // Integer window-absolute pixel coordinates
    glm::vec2 _mousePos;

    // Mouse speed multipliers
    float _mouseScrollMult;
    float _mousePosMult;
  };
} // dh::vis