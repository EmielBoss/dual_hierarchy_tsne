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
    EmbeddingRenderTask(sne::MinimizationBuffers minimizationBuffers, sne::Params* params, std::vector<GLuint> classTextures, std::vector<uint> classCounts, int priority);
    ~EmbeddingRenderTask();

    // Copy constr/assignment is explicitly deleted
    EmbeddingRenderTask(const EmbeddingRenderTask&) = delete;
    EmbeddingRenderTask& operator=(const EmbeddingRenderTask&) = delete;

    // Move constr/operator moves handles
    EmbeddingRenderTask(EmbeddingRenderTask&&) noexcept;
    EmbeddingRenderTask& operator=(EmbeddingRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj) override;
    void drawImGuiComponent() override;
    void drawImGuiComponentSecondary() override;
    
    void generateClassColors();
    void createVAO();
    void setClass(int i);
    void unsetClass(int i);

    void setPointRadius(float pointRadius) { _pointRadius = pointRadius; }
    std::set<int> getClassesSet() { return _classesSet; }
    bool getSetChanged() { return _setChanged; }
    void setNumSelectedNeighbors(uint nSelectedNeighbors) { _nSelectedNeighbors = nSelectedNeighbors; }
    uint getColorMapping() { return _colorMapping; }
    bool getWeighForces() { return _weighForces; }
    void setWeighForces(bool weighForces) { _weighForces = weighForces; }
    float getWeightFixed() { return _weightFixed; }
    float getWeightFalloff() { return _weightFalloff; }
    float getPerplexity() { return _perplexity; }
    bool getFocusButtonPressed() { return _focusButtonPressed; }
    int getClassButtonPressed() { return _classButtonPressed; }
    uint getK() { return (uint) _k; }
    void setSelectionMode(bool selectLabeledOnly) { _selectLabeledOnly = selectLabeledOnly; }
    void setMinimizationBuffers(sne::MinimizationBuffers minimizationBuffers) {
      _minimizationBuffers = minimizationBuffers;
      createVAO(); // Recreate VAO with updated embedding buffer
    }

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::Params* _params;

    // ImGui state
    bool _weighForces;
    float _weightFixed;
    float _weightFalloff;
    int _numClusters;
    int _numClustersPrev;
    bool _selectLabeledOnly;
    uint _colorMapping;
    float _pointRadius;
    float _pointOpacity;
    float _perplexity;
    int _k;
    bool _focusButtonPressed;
    int _classButtonPressed;
    std::vector<glm::vec4> _colors;
    std::vector<uint> _classCounts;
    std::vector<std::string> _classNames;
    std::vector<bool> _classIsSet;
    std::set<int> _classesSet;
    bool _setChanged;
    uint _nSelectedNeighbors;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    sne::MinimizationBuffers _minimizationBuffers;
    std::vector<GLuint> _classTextures;
    GLuint _colorBuffer;
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
      swap(a._weighForces, b._weighForces);
      swap(a._weightFixed, b._weightFixed);
      swap(a._weightFalloff, b._weightFalloff);
      swap(a._numClusters, b._numClusters);
      swap(a._numClustersPrev, b._numClustersPrev);
      swap(a._colorMapping, b._colorMapping);
      swap(a._pointRadius, b._pointRadius);
      swap(a._pointOpacity, b._pointOpacity);
      swap(a._perplexity, b._perplexity);
      swap(a._k, b._k);
      swap(a._focusButtonPressed, b._focusButtonPressed);
      swap(a._classButtonPressed, b._classButtonPressed);
      swap(a._colors, b._colors);
      swap(a._buffers, b._buffers);
      swap(a._classTextures, b._classTextures);
      swap(a._classCounts, b._classCounts);
      swap(a._classNames, b._classNames);
      swap(a._classIsSet, b._classIsSet);
      swap(a._classesSet, b._classesSet);
      swap(a._setChanged, b._setChanged);
      swap(a._nSelectedNeighbors, b._nSelectedNeighbors);
      swap(a._colorBuffer, b._colorBuffer);
      swap(a._vaoHandle, b._vaoHandle);
      swap(a._program, b._program);
    }
  };
} // dh::vis