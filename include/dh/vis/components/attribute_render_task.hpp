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

#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/vis/render_queue.hpp"
#include "dh/vis/components/selection_input_task.hpp"

namespace dh::vis {
  class AttributeRenderTask : public RenderTask {

  public:
    AttributeRenderTask();
    AttributeRenderTask(sne::Params* params, int priority, sne::MinimizationBuffers minimizationBuffers, sne::SimilaritiesBuffers similaritiesBuffers, GLuint colorBuffer, const int* labelPtr);
    ~AttributeRenderTask();

    // Copy constr/assignment is explicitly deleted
    AttributeRenderTask(const AttributeRenderTask&) = delete;
    AttributeRenderTask& operator=(const AttributeRenderTask&) = delete;

    // Move constr/operator moves handles
    AttributeRenderTask(AttributeRenderTask&&) noexcept;
    AttributeRenderTask& operator=(AttributeRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj) override;
    void drawImGuiComponent() override;
    void drawImGuiComponentSecondary() override;

    void drawImGuiTab(uint selectionIndex, uint typeIndex, const char* text);
    void drawImGuiTexture();
    void drawImPlotBarPlot();

    float getBufferValue(GLuint buffer, int index);
    void setOverlayTexel(int texelIndex, std::vector<float> color = {1.f, 1.f, 1.f, 1.f});
    void mirrorWeightsToOverlay();
    void brushTexels(uint centerTexelIndex, int radius, float weight);
    void eraseTexels(uint centerTexelIndex, int radius);
    void setTexelWeight(uint attributeIndex, float weight);
    float getTexelWeight(uint texelIndex);
    float getTexelValue(uint texelIndex, GLuint buffer);
    void autoweighAttributes(uint textureType, float percentage);
    void clearAttributeWeights();
    void invertAttributeWeights();
    void refineAttributeWeights(uint textureType);
    void setClass(int i);
    void unsetClass(int i);
    void update(std::vector<uint> selectionCounts);
    void clear();
    uint currentTabIndex() { return _currentTabUpper * 2 + _currentTabLower; }
    void addArchetype(uint archetypeClass);
    void assess(uint symmetricSize);
    float sumWeightedAttributeValues(uint index);
    
  private:
    enum class BufferType {
      ePairwiseAttrDists,

      Length
    };

    enum class TextureType {
      eAveragePrimary,
      eVariancePrimary,
      eAverageSecondary,
      eVarianceSecondary,
      eAverageDifference,
      eVarianceDifference,

      ePairwiseDiffsAll,
      ePairwiseDiffsInter,
      ePairwiseDiffsIntra,

      eOverlay,

      Length
    };

    enum class ProgramType {
      ePairwiseAttrDiffsNeiComp,
      ePairwiseAttrDiffsAllComp,
      dGuessClasses,

      Length
    };

    // State
    bool _isInit;
    sne::Params* _params;
    dh::vis::Input _input;

    // ImGui state
    std::vector<uint> _selectionCounts;
    float _similarityWeight;
    float _attributeWeight;
    int _brushRadius;
    float _autoselectPercentage;
    uint _buttonPressed;
    uint _currentTabUpper;
    uint _currentTabLower;
    uint _previousTabIndex;
    int _draggedTexel;
    int _draggedTexelPrev;
    std::set<uint> _weightedAttributeIndices;
    int _classButtonPressed;
    std::vector<uint> _classCounts;
    std::vector<uint> _classCountsSelected;
    std::vector<std::string> _classNames;
    std::vector<bool> _classIsSet;
    std::set<int> _classesSet;
    bool _setChanged;
    std::vector<glm::vec3> _classColors;
    bool _vizAllPairs;
    std::vector<uint> _denominators;
    std::vector<uint> _archetypeClasses; // Same order as _buffersTextureDataArchetypes

    // Objects
    sne::MinimizationBuffers _minimizationBuffers;
    sne::SimilaritiesBuffers _similaritiesBuffers;
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<TextureType, GLuint> _buffersTextureData;
    std::vector<GLuint> _buffersTextureDataArchetypes;
    util::EnumArray<TextureType, GLuint> _textures;
    std::vector<GLuint> _texturesArchetypes;
    std::vector<GLuint> _classTextures;
    GLuint _bufferClassColors;
    util::EnumArray<ProgramType, util::GLProgram> _programs;

  public:
    bool isInit() const { return _isInit; }

    std::set<uint>& getWeightedAttributeIndices() { return _weightedAttributeIndices; }
    uint getButtonPressed() { return _buttonPressed; }
    float getSimilarityWeight() { return _similarityWeight; }
    void setSelectionCounts(std::vector<uint> selectionCounts) { _selectionCounts = selectionCounts; }
    void setInput(dh::vis::Input input) { _input = input; }
    int getClassButtonPressed() { return _classButtonPressed; }
    std::vector<uint> getArchetypeClasses() { return _archetypeClasses; }
    void setArchetypeClasses(std::vector<uint> archetypeClasses) { _archetypeClasses = archetypeClasses; }
    std::vector<GLuint> getArchetypeHandles() { return _buffersTextureDataArchetypes; }
    void setArchetypeHandles(std::vector<GLuint> archetypeHandles) { _buffersTextureDataArchetypes = archetypeHandles; }
    void setMinimizationBuffers(sne::MinimizationBuffers minimizationBuffers) { _minimizationBuffers = minimizationBuffers; }
    void setSimilaritiesBuffers(sne::SimilaritiesBuffers similaritiesBuffers) { _similaritiesBuffers = similaritiesBuffers; }

    // std::swap impl
    friend void swap(AttributeRenderTask& a, AttributeRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._input, b._input);
      swap(a._selectionCounts, b._selectionCounts);
      swap(a._draggedTexel, b._draggedTexel);
      swap(a._buttonPressed, b._buttonPressed);
      swap(a._attributeWeight, b._attributeWeight);
      swap(a._weightedAttributeIndices, b._weightedAttributeIndices);
      swap(a._brushRadius, b._brushRadius);
      swap(a._similarityWeight, b._similarityWeight);
      swap(a._autoselectPercentage, b._autoselectPercentage);
      swap(a._vizAllPairs, b._vizAllPairs);
      swap(a._denominators, b._denominators);
      swap(a._currentTabUpper, b._currentTabUpper);
      swap(a._currentTabLower, b._currentTabLower);
      swap(a._classTextures, b._classTextures);
      swap(a._classCounts, b._classCounts);
      swap(a._classCountsSelected, b._classCountsSelected);
      swap(a._classNames, b._classNames);
      swap(a._classIsSet, b._classIsSet);
      swap(a._classesSet, b._classesSet);
      swap(a._setChanged, b._setChanged);
      swap(a._classButtonPressed, b._classButtonPressed);
      swap(a._archetypeClasses, b._archetypeClasses);
      swap(a._buffers, b._buffers);
      swap(a._buffersTextureData, b._buffersTextureData);
      swap(a._buffersTextureDataArchetypes, b._buffersTextureDataArchetypes);
      swap(a._textures, b._textures);
      swap(a._texturesArchetypes, b._texturesArchetypes);
      swap(a._classColors, b._classColors);
      swap(a._bufferClassColors, b._bufferClassColors);
      swap(a._minimizationBuffers, b._minimizationBuffers);
      swap(a._similaritiesBuffers, b._similaritiesBuffers);
      swap(a._programs, b._programs);
    }
  };
} // dh::vis