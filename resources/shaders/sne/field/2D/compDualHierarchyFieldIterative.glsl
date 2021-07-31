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

// Necessary extensions (nvidia-centric)
#extension GL_NV_shader_atomic_float : require        // atomicAdd(f32) support
#extension GL_KHR_shader_subgroup_clustered : require // subgroupClusteredAdd(...) support
#extension GL_KHR_shader_subgroup_shuffle : require   // subgroupShuffle(...) support

// Wrapper structure for pair queue data
struct Pair {
  uint f;       // Field hierarchy node index
  uint e;       // Embedding hierarchy node index
};

// Wrapper structure for bounding box data
struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
layout(binding = 2, std430) restrict readonly buffer EPosi { vec2 ePosBuffer[]; };
layout(binding = 3, std430) restrict readonly buffer FNode { uint fNodeBuffer[]; };
layout(binding = 4, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; };  // Atomic
layout(binding = 5, std430) restrict readonly buffer IQueu { Pair iQueueBuffer[]; };   // Input
layout(binding = 6, std430) restrict readonly buffer IQHea { uint iQueueHead; };       // Atomic
layout(binding = 7, std430) restrict writeonly buffer OQue { Pair oQueueBuffer[]; };   // Output
layout(binding = 8, std430) restrict readonly buffer OQHea { uint oQueueHead; };       // Atomic
layout(binding = 9, std430) restrict writeonly buffer LQue { Pair lQueueBuffer[]; };   // Leaf
layout(binding = 10, std430) restrict coherent buffer LQHe { uint lQueueHead; };       // Atomic
layout(binding = 11, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
layout(binding = 12, std430) restrict writeonly buffer RQu { Pair rQueueBuffer[]; };   // Atomic
layout(binding = 13, std430) restrict coherent buffer RQHe { uint rQueueHead; };       // Atomic

// Uniform values
layout(location = 0) uniform uint dhLvl;    // Current level in virtual dual hierarchy
layout(location = 1) uniform uint eLvls;    // Total nr. of levels in embedding hierarchy
layout(location = 2) uniform uint fLvls;    // Total nr. of levels in field hierarchy
layout(location = 3) uniform float theta2;  // Squared approximation param. for Barnes-Hut
layout(location = 4) uniform bool doBhCrit; // Use new Barnes-Hut criterion

// Shared memory
shared Bounds bounds;

// Constants
#define BVH_KNODE_2D 4
#define BVH_LOGK_2D 2
#define BVH_KLEAF_2D 4
#define BVH_LARGE_LEAF 16

// Invoc shuffle constants
const uint thread = gl_LocalInvocationID.x % BVH_KNODE_2D;
const uint base = gl_LocalInvocationID.x - thread;
const uint rotate = base + (thread + 1) % BVH_KNODE_2D;

// Compute dot product of vector with itself, ie. squared eucl. distance to vector from origin
float sdot(vec2 v) {
  return dot(v, v);
}

uint shrinkBits15(uint i) {
  i = i & 0x55555555u;
  i = (i | (i >> 1u)) & 0x33333333u;
  i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
  i = (i | (i >> 4u)) & 0x00FF00FFu;
  i = (i | (i >> 8u)) & 0x0000FFFFu;
  return i;
}

uvec2 decode(uint i) {
  uint x = shrinkBits15(i);
  uint y = shrinkBits15(i >> 1);
  return uvec2(x, y);
}

void main() {
  const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x)
                    / BVH_KNODE_2D;
  if (idx >= iQueueHead) {
    return;
  }

  // Load bounds buffer into shared memory
  if (gl_LocalInvocationID.x == 0) {
    bounds = boundsBuffer;
  }
  barrier();

  // Read node pair indices once, share with subgroup operations, then subdivide each side
  Pair pair;
  if (thread == 0) {
    pair = iQueueBuffer[idx];
  }
  pair.f = subgroupShuffle(pair.f, base) * BVH_KNODE_2D + 1 + thread;
  pair.e = subgroupShuffle(pair.e, base) * BVH_KNODE_2D + 1 + thread;

  // Load embedding/field node data. Field node is const, emb node gets shuffled over subgroup
  const uint fNode = fNodeBuffer[pair.f];   // packed as: 30 bits leaf addr, 2 bits node type
  vec4 eNode0 = eNode0Buffer[pair.e];       // packed as: bbox c.o.m. (xy(z)), range extent (w)
  vec4 eNode1 = eNode1Buffer[pair.e];       // packed as: bbox extent (xy(z)), range min    (w)

  // Move down singleton chain to leaf node if necessary
  if ((fNode & 3u) == 1u) {
    pair.f = fNode >> 2u;
  }

  // Determine fNode bbox and center implicitly
  const uint fLvl = (fNode & 3u )== 1u ? fLvls - 1 : dhLvl;
  const uvec2 px = decode(pair.f - (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl)));
  const vec2 fbbox = bounds.range / vec2(1u << fLvl);
  const vec2 fpos = bounds.min + fbbox * (vec2(px) + 0.5);
  
  // Values accumulated or kept constant during rotation
  const uint pairBase = pair.e - thread;
  const bool fIsLeaf = (fNode & 3u) == 1u;
  const uint eOffset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (eLvls - 1));
  vec3 field = vec3(0);

  // Compare node pairs BVH_KNODE_2D times, rotating embedding nodes through the subgroup
  for (uint i = 0; i < BVH_KNODE_2D; ++i) {
    // Rotate embedding node data over BVH_KNODE_2D invocs in the subgroup
    eNode0 = subgroupShuffle(eNode0, rotate);
    eNode1 = subgroupShuffle(eNode1, rotate);

    // Prevent computation for dead node pairs. Yay divergence
    if (eNode0.w == 0 || fNode == 0) {
      continue;
    }
    
    // Compute vector dist. and squared eucl dist. between nodes
    const vec2 t = fpos - eNode0.xy;
    const float t2 = dot(t, t);

    // Barnes-Hut criterion
    float r2;
    if (doBhCrit) {
      // Reflect vector so it always approaches almost-orthogonal to the diameter
      const vec2 b = normalize(abs(t) * vec2(-1, 1));
      // Compute relative squared diams
      r2 = max(
        sdot(eNode1.xy - b * dot(eNode1.xy, b)),
        sdot(fbbox - b * dot(fbbox, b))
      );
    } else {
      // Old Barnes-Hut criterion
      r2 = max(sdot(eNode1.xy), sdot(fbbox));
    }
    
    // More rotate data
    const uint rot_e = pairBase + (thread + i + 1) % BVH_KNODE_2D;
    const bool eIsLeaf = eNode0.w <= BVH_KLEAF_2D || rot_e >= eOffset;

    // Barnes-Hut test
    if (r2 / t2 < theta2) {
      // Approximation passes, compute approximated value and truncate
      const float tStud = 1.f / (1.f + t2);
      field += eNode0.w * vec3(tStud, t * (tStud * tStud));
    } else if (!eIsLeaf && !fIsLeaf) {
      // Push pair on work queue for further subdivision
      oQueueBuffer[atomicAdd(oQueueHead, 1)] = Pair(pair.f, rot_e);
    } else if (!fIsLeaf) {
      // Push pair on secondary work queue for further subdivision
      rQueueBuffer[atomicAdd(rQueueHead, 1)] = Pair(pair.f, rot_e);
    } else if (eNode0.w > BVH_LARGE_LEAF) {
      // Optimize large leaves in separate shader
      lQueueBuffer[atomicAdd(lQueueHead, 1)] = Pair(pair.f, rot_e);
    } else {
      // Compute small leaves directly
      const uint begin = uint(eNode1.w);
      const uint end = begin + uint(eNode0.w);
      for (uint j = begin; j < end; ++j) {
        const vec2 t = fpos - ePosBuffer[j];
        const float tStud = 1.f / (1.f + dot(t, t));
        field += vec3(tStud, t * (tStud * tStud));
      }
    }
  }
  
  // Add computed forces to field hierarchy
  if (field != vec3(0)) {
    const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
    atomicAdd(fFieldBuffer[addr.x], field.x);
    atomicAdd(fFieldBuffer[addr.y], field.y);
    atomicAdd(fFieldBuffer[addr.z], field.z);
  }
}