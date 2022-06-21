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
  class TrackballInputTask : public InputTask {
  public:
    TrackballInputTask();

    glm::mat4 rotateLookat(glm::vec2 v);
    void process() override;
    void mousePosInput(double xPos, double yPos) override;
    void mouseButtonInput(int button, int action) override;
    void mouseScrollInput(double xScroll, double yScroll) override;
    void keyboardInput(int button, int action) override;

    glm::mat4 matrix() const { return _matrix; }
    glm::mat3 changeOfBaseToCam() const { return _changeOfBaseToCam; }

  private:
    // State
    bool _mouseTrackState;
    float _mouseScrollState;
    glm::vec2 _mousePosState;
    glm::vec2 _mousePosStatePrev;
    glm::mat4 _lookatState;
    glm::mat4 _matrix;
    bool _shiftPressed;
    float _timePrev; // Time of last frame
    glm::vec4 _cameraPos;
    glm::vec4 _up;
    glm::mat3 _changeOfBaseToCam;

    // Mouse speed multipliers
    float _mouseScrollMult;
    float _mousePosMult;
  };
} // dh::vis