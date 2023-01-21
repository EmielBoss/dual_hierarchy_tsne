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

#include <algorithm>
#include <vector>
#include <set>
#include "dh/types.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/timer.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/util/cu/timer.cuh"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/util/gl/window.hpp" //

namespace dh::sne {
  class Similarities {
  public:
    // Constr/destr
    Similarities();
    Similarities(const float* dataPtr, Params* params);
    ~Similarities();

    // Copy constr/assignment is explicitly deleted
    Similarities(const Similarities&) = delete;
    Similarities& operator=(const Similarities&) = delete;

    // Move constr/operator moves handles
    Similarities(Similarities&&) noexcept;
    Similarities& operator=(Similarities&&) noexcept;

    float average(std::vector<float> vec);
    void displayHistogram(std::vector<float> inter, std::vector<float> intra, bool relative);
    void displayBarplot(std::vector<float> ys);

    // Compute similarities
    void comp();
    void recomp(GLuint selectionBufferHandle, float perplexity, uint k);
    void weighSimilarities(float weight, GLuint selectionBufferHandle = 0, bool interOnly = false);
    void weighSimilaritiesPerAttribute(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, GLuint labelsBufferHandle);
    void reset();

  private:
    enum class BufferType {
      eDataset,
      eDistancesL1,
      eNeighbors,
      eSimilarities,
      eSimilaritiesOriginal,
      eLayout,
      eAttributeWeights,
      
      Length
    };

    // Basically buffers that have to be destroyed and recreated in order to grow their memory allocation
    enum class BufferTempType {
      eDistances,
      eNeighbors,
      eSimilarities,
      eSizes,
      eScan,
      eCounts,
      eWeightedAttributeIndices,

      Length
    };

    enum class ProgramType {
      eSimilaritiesComp,
      eExpandComp,
      eLayoutComp,
      eNeighborsComp,
      eNeighborsSortComp,
      eL1DistancesComp,
      eWeighSimilaritiesComp,
      eWeighSimilaritiesPerAttributeComp,
      
      Length
    };

    enum class TimerType {
      eSimilaritiesComp,
      eExpandComp,
      eLayoutComp,
      eNeighborsComp,
      
      Length
    };

    // State
    bool _isInit;
    Params* _params;
    const float* _dataPtr;
    uint _kPrev;
    uint _symmetricSize;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<BufferTempType, GLuint> _buffersTemp;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;
    util::CUTimer _knnTimer;
  
  public:
    // Getters
    bool isInit() const { return _isInit; }
    SimilaritiesBuffers buffers() const {
      return {
        _buffers(BufferType::eDataset),
        _buffers(BufferType::eSimilarities),
        _buffers(BufferType::eLayout),
        _buffers(BufferType::eNeighbors),
        _buffers(BufferType::eAttributeWeights)
      };
    }

    // std::swap impl
    friend void swap(Similarities& a, Similarities& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._dataPtr, b._dataPtr);
      swap(a._kPrev, b._kPrev);
      swap(a._buffers, b._buffers);
      swap(a._buffersTemp, b._buffersTemp);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._knnTimer, b._knnTimer);
    }
  };
} // dh::sne