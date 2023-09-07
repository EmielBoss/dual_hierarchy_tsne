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

#version 460 core

struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

vec3 grey = vec3(150.f, 150.f, 150.f);

// Input attributes
layout(location = 0) in uint elementIn;

// Output attributes
layout(location = 0) out vec4 colorOut;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer EmbRel { vec2 embRelBuffer[]; };
layout(binding = 1, std430) restrict readonly buffer SimRel { float simRelBuffer[]; };

// Uniform locations
layout(location = 0) uniform mat4 model_view;
layout(location = 1) uniform mat4 proj;
layout(location = 2) uniform float linkOpacity;
layout(location = 3) uniform uint colorMapping;

void main() {
  // Calculate vertex position
  vec2 positionRel = embRelBuffer[elementIn];
  positionRel.y = 1.f - positionRel.y;
  gl_Position = proj * model_view * vec4(positionRel, 0, 1);

  // Calculate vertex/link color
  float simRel = simRelBuffer[gl_VertexID / 2];
  if(colorMapping == 0) {
    colorOut = vec4(0.f, 0.f, 0.f, 1.f);
  } else
  if(colorMapping == 1) {
      colorOut = vec4(simRel, 0.f, 1 - simRel, 1.f);
  } else
  if(colorMapping == 2) {
      colorOut = vec4(0.f, 0.f, 0.f, simRel);
  } else
  if(colorMapping == 3) {
    colorOut = vec4(simRel, 0.f, 1 - simRel, simRel);
  }

  colorOut.a *= linkOpacity;
    
}