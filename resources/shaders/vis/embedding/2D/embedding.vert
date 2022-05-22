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

// 10 distinguishable label colors
const vec3 colors[10] = vec3[10](
  vec3(16, 78, 139),
  vec3(139, 90, 43),
  vec3(138, 43, 226),
  vec3(0, 128, 0),
  vec3(255, 150, 0),
  vec3(204, 40, 40),
  vec3(131, 139, 131),
  vec3(0, 205, 0),
  // vec3(235, 235, 235),
  vec3(20, 20, 20),
  vec3(0, 150, 255)
);

// Input attributes
layout(location = 0) in vec2 positionIn;
layout(location = 1) in vec2 embeddingIn;

// Output attributes
layout(location = 0) out vec2 embeddingOut;
layout(location = 1) out vec2 fragEmbeddingOut;
layout(location = 2) out vec4 colorOut;
layout(location = 3) out float multiplier;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
layout(binding = 1, std430) restrict readonly buffer LabelsBuffer { int labels[]; };
layout(binding = 2, std430) restrict readonly buffer LabeledBuffer { uint labeled[]; };
layout(binding = 3, std430) restrict readonly buffer SelectedBuffer { uint selected[]; };
layout(binding = 4, std430) restrict readonly buffer NeighborhoodPreservationBuffer { float neighborhoodPreservation[]; };

// Uniform locations
layout(location = 0) uniform mat4 model_view;
layout(location = 1) uniform mat4 proj;
layout(location = 2) uniform float pointOpacity;
layout(location = 3) uniform float pointRadius;
layout(location = 4) uniform uint colorMapping;
layout(location = 5) uniform bool canDrawLabels;
layout(location = 6) uniform bool selectLabeledOnly;

void main() {
  multiplier = selectLabeledOnly && labeled[gl_InstanceID] == 1 ? 2.f : 1.f;
  float divider = selectLabeledOnly && labeled[gl_InstanceID] == 0 ? 30.f : 1.f;

  // Calculate embedding position, fragment position
  embeddingOut = (embeddingIn - bounds.min) * bounds.invRange;
  embeddingOut.y = 1.f - embeddingOut.y;
  fragEmbeddingOut = embeddingOut + positionIn * pointRadius * multiplier;

  // Calculate vertex position
  gl_Position = proj * model_view * vec4(fragEmbeddingOut, 0, 1);


  // Calculate output color depending on color mapping, label and whether it is selected, whether to even draw labels
  vec3 color;
  const int label = labels[gl_InstanceID];
  if(colorMapping == 1) { // Labels
    int colorIndex = canDrawLabels && label >= 0 ? label : 9;
    color = colors[colorIndex];
  }
  else if(colorMapping == 2) { // Neighborhood preservation
    float value = neighborhoodPreservation[gl_InstanceID];
    color = vec3(255, (1-value) * 255, (1-value) * 200);
  }
  else {
    color = colors[9];
  }

  if(selected[gl_InstanceID] == 1) {
    colorOut = vec4(color / 355.0f, pointOpacity);
  } else {
    colorOut = vec4(color / 255.0f, pointOpacity / divider);
  }
}