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
#include <fstream>
#include <stdexcept>
#include <vector>
#include <set>
#include "dh/util/debug.hpp"
#include "dh/util/io.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::util {
  void indicateLabeled(const int* labelPtr, uint n, uint nClasses, uint nLabeledPerClass, GLuint bufferLabeled)
  {
    std::vector<int> labeled(n, 0);
    
    for(uint c = 0; c < nClasses; ++c) {
      uint count = 0;
      for(uint i = 0; i < n && count < nLabeledPerClass; ++i) {
        if(*(labelPtr + i) == c) {
          labeled[i] = 1;
          count++;
        }
      }
    }

    glNamedBufferStorage(bufferLabeled, n * sizeof(int), labeled.data(), 0);
  }

  void readState(uint n, uint nHighDims, uint d, std::array<GLuint, 24> buffers, GLuint& bufferAttributeWeights, std::set<uint>& weightedAttributeIndices, std::vector<GLuint>& archetypeHandles, std::vector<uint>& archetypeLabels, std::vector<uint>& archetypeIndices)
  {
    glDeleteBuffers(archetypeHandles.size(), archetypeHandles.data());
    // dh::util::readGLBuffer<float>(buffers[20], "rel"); // eEmbeddingRelative
    dh::util::readGLBuffer<uint>(buffers[16], "fxd"); // eFixed
    dh::util::readGLBuffer<uint>(buffers[22], "dsb"); // eDisabled
    dh::util::readGLBuffer<float>(buffers[18], "wgt"); // eWeights
    dh::util::readGLBuffer<uint>(buffers[15], "slc"); // eSelection
    dh::util::readGLBuffer<uint>(buffers[19], "lbd"); // eLabeled
    dh::util::readGLBuffer<float>(bufferAttributeWeights, "awt"); // eAttributeWeights
    weightedAttributeIndices = dh::util::readSet<uint>("wai"); // Weighted attribute indices

    if(!std::ifstream("./buffer_dumps/ats.txt").good()) { return; } // No archetypes files found
    archetypeLabels = dh::util::readVector<uint>("atc");
    archetypeIndices = dh::util::readVector<uint>("ati");
    uint nArchetypes = archetypeLabels.size();
    GLuint tempBuffer;
    glCreateBuffers(1, &tempBuffer);
    glNamedBufferStorage(tempBuffer, nArchetypes * nHighDims * sizeof(float), nullptr, 0);
    dh::util::readGLBuffer<float>(tempBuffer, "ats");
    archetypeHandles = std::vector<GLuint>(nArchetypes);
    glCreateBuffers(archetypeHandles.size(), archetypeHandles.data());
    for(uint i = 0; i < nArchetypes; ++i) {
      glNamedBufferStorage(archetypeHandles[i], nHighDims * sizeof(float), nullptr, 0);
      glCopyNamedBufferSubData(tempBuffer, archetypeHandles[i], i * nHighDims * sizeof(float), 0, nHighDims * sizeof(float));
    }
    glDeleteBuffers(1, &tempBuffer);
    glAssert();
  }

  void writeState(uint n, uint nHighDims, uint d, std::array<GLuint, 24> buffers, GLuint bufferAttributeWeights, std::set<uint> weightedAttributeIndices, std::vector<GLuint> archetypeHandles, std::vector<uint> archetypeLabels, std::vector<uint>& archetypeIndices)
  {
    // dh::util::writeGLBuffer<float>(buffers[20], n, d, "rel"); // eEmbeddingRelative
    dh::util::writeGLBuffer<int>(buffers[16], n, 1, "fxd"); // eFixed
    dh::util::writeGLBuffer<int>(buffers[22], n, 1, "dsb"); // eDisabled
    dh::util::writeGLBuffer<float>(buffers[18], n, 1, "wgt"); // eWeights
    dh::util::writeGLBuffer<int>(buffers[15], n, 1, "slc"); // eSelection
    dh::util::writeGLBuffer<int>(buffers[19], n, 1, "lbd"); // eLabeled
    dh::util::writeGLBuffer<float>(bufferAttributeWeights, nHighDims, 1, "awt"); // eAttributeWeights
    dh::util::writeSet<uint>(weightedAttributeIndices, "wai"); // Weighted attribute indices
    
    uint nArchetypes = archetypeHandles.size();
    if(nArchetypes == 0) { return; }
    GLuint tempBuffer;
    glCreateBuffers(1, &tempBuffer);
    glNamedBufferStorage(tempBuffer, nArchetypes * nHighDims * sizeof(float), nullptr, 0);
    for(uint i = 0; i < nArchetypes; ++i) {
      glCopyNamedBufferSubData(archetypeHandles[i], tempBuffer, 0, i * nHighDims * sizeof(float), nHighDims * sizeof(float));
    }
    dh::util::writeGLBuffer<float>(tempBuffer, nArchetypes, nHighDims, "ats");
    dh::util::writeVector<uint>(archetypeLabels, nArchetypes, 1, "atc");
    dh::util::writeVector<uint>(archetypeIndices, nArchetypes, 1, "ati");
    glDeleteBuffers(1, &tempBuffer);
    glAssert();
  }

} // dh::util