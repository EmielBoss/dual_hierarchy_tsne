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

#include <chrono>
#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/vis/render_queue.hpp"

namespace dh::vis {
  template <uint D>
  class EmbeddingRenderTask : public RenderTask {
    using vec = util::AlignedVec<D, float>;

  public:
    EmbeddingRenderTask();
    EmbeddingRenderTask(sne::MinimizationBuffers minimizationBuffers, sne::Params params, int priority);
    ~EmbeddingRenderTask();

    // Copy constr/assignment is explicitly deleted
    EmbeddingRenderTask(const EmbeddingRenderTask&) = delete;
    EmbeddingRenderTask& operator=(const EmbeddingRenderTask&) = delete;

    // Move constr/operator moves handles
    EmbeddingRenderTask(EmbeddingRenderTask&&) noexcept;
    EmbeddingRenderTask& operator=(EmbeddingRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle = 0) override;
    void drawImGuiComponent() override;
    uint getColorMapping() { return _colorMapping; }
    void setColorMapping(uint colorMapping) { _colorMapping = colorMapping; }
    bool getWeighForces() { return _weighForces; }
    void setWeighForces(bool weighForces) { _weighForces = weighForces; }
    float getWeightFixed() { return _weightFixed; }
    float getWeightFalloff() { return _weightFalloff; }
    void setSelectionMode(bool selectLabeledOnly) { _selectLabeledOnly = selectLabeledOnly; }

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::MinimizationBuffers _minimizationBuffers;
    sne::Params _params;

    // ImGui state
    bool _canDrawLabels;
    bool _weighForces;
    float _weightFixed;
    float _weightFalloff;
    int _numClusters;
    int _numClustersPrev;
    bool _selectLabeledOnly;
    uint _colorMapping;
    float _pointRadius;
    float _pointOpacity;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }

    enum ColorMapping {
      none,
      labels,
      neighborhoodPreservation
    };
    
    // std::swap impl
    friend void swap(EmbeddingRenderTask& a, EmbeddingRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._minimizationBuffers, b._minimizationBuffers);
      swap(a._params, b._params);
      swap(a._canDrawLabels, b._canDrawLabels);
      swap(a._weighForces, b._weighForces);
      swap(a._weightFixed, b._weightFixed);
      swap(a._weightFalloff, b._weightFalloff);
      swap(a._numClusters, b._numClusters);
      swap(a._numClustersPrev, b._numClustersPrev);
      swap(a._colorMapping, b._colorMapping);
      swap(a._pointRadius, b._pointRadius);
      swap(a._pointOpacity, b._pointOpacity);
      swap(a._buffers, b._buffers);
      swap(a._vaoHandle, b._vaoHandle);
      swap(a._program, b._program);
    }
  };
} // dh::vis