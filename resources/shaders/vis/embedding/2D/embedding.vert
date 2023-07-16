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
layout(location = 0) in vec2 positionIn;
layout(location = 1) in vec2 embeddingRelIn;

// Output attributes
layout(location = 0) out vec2 embeddingRelOut;
layout(location = 1) out vec2 fragEmbeddingOut;
layout(location = 2) out vec4 colorOut;
layout(location = 3) out float multiplier;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
layout(binding = 1, std430) restrict readonly buffer LabelsBuffer { int labels[]; };
layout(binding = 2, std430) restrict readonly buffer LabeledBuffer { uint labeled[]; };
layout(binding = 3, std430) restrict readonly buffer DisabledBuffer { uint disabled[]; };
layout(binding = 4, std430) restrict readonly buffer FixedBuffer { uint fixxed[]; };
layout(binding = 5, std430) restrict readonly buffer SelectionBuffer { uint selection[]; };
layout(binding = 6, std430) restrict readonly buffer NeighborhoodPreservationBuffer { float neighborhoodPreservation[]; };
layout(binding = 7, std430) restrict readonly buffer ClassColorBuffer { vec3 classColors[]; };
layout(binding = 8, std430) restrict readonly buffer PointColorBuffer { vec3 pointColors[]; };

// Uniform locations
layout(location = 0) uniform mat4 model_view;
layout(location = 1) uniform mat4 proj;
layout(location = 2) uniform float pointOpacity;
layout(location = 3) uniform float pointRadius;
layout(location = 4) uniform uint colorMapping;
layout(location = 5) uniform bool canDrawLabels;
layout(location = 6) uniform bool selectLabeledOnly;
layout(location = 7) uniform float divisor;

void main() {

  multiplier = selectLabeledOnly && labeled[gl_InstanceID] == 1 ? 5.f : 1.f;
  multiplier += multiplier * 0.5f * fixxed[gl_InstanceID];
  float divider = selectLabeledOnly && labeled[gl_InstanceID] == 0 ? 5.f : 1.f;

  // Calculate embedding position, fragment position
  embeddingRelOut = embeddingRelIn;
  embeddingRelOut.y = 1.f - embeddingRelOut.y;
  fragEmbeddingOut = embeddingRelOut + positionIn * pointRadius * multiplier;

  // Calculate vertex position
  gl_Position = proj * model_view * vec4(fragEmbeddingOut, 0, 1);
  if(disabled[gl_InstanceID] == 1) { gl_Position = vec4(1000.f, 1000.f, 1000.f, 1.f); }
  if(selectLabeledOnly && labeled[gl_InstanceID] == 1) { gl_Position.z = 0.1f; }

  if(colorMapping == 0) { // None
    vec3 color = grey;
    if(!selectLabeledOnly) {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 200.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 400.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    } else {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 255.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 400.f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    }
  } else
  if(colorMapping == 1) { // Labels
    vec3 color = classColors[canDrawLabels && labels[gl_InstanceID] >= 0 ? labels[gl_InstanceID] : 0];
    if(!selectLabeledOnly) {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 400.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 200.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    } else {
      if(selection[gl_InstanceID] == 0 && labeled[gl_InstanceID] == 0) { colorOut = vec4(grey / 400.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 0 && labeled[gl_InstanceID] == 1) { colorOut = vec4(color / 400.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1 && labeled[gl_InstanceID] == 0) { colorOut = vec4(grey / 255.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 1 && labeled[gl_InstanceID] == 1) { colorOut = vec4(color / 255.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    }
  } else
  if(colorMapping == 2) { // Colors
    vec3 color = pointColors[gl_InstanceID];
    if(!selectLabeledOnly) {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 255.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(0.f, 0.f, 0.f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    } else {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 255.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 400.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    }
  } else
  if(colorMapping == 3) { // Neighborhood preservation
    float value = neighborhoodPreservation[gl_InstanceID];
    vec3 color = vec3(255, (1-value) * 255, (1-value) * 200);
    if(!selectLabeledOnly) {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 255.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 400.0f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    } else {
      if(selection[gl_InstanceID] == 0) { colorOut = vec4(color / 255.0f, pointOpacity / divider); } else
      if(selection[gl_InstanceID] == 1) { colorOut = vec4(color / 400.f, pointOpacity); } else
      if(selection[gl_InstanceID] == 2) { colorOut = vec4(color / divisor, pointOpacity); }
    }
  }

}