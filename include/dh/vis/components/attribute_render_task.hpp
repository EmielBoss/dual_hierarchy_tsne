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
#include <imgui.h>
#include <implot.h>

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

    void drawImGuiTab(uint tabIndex, const char* text);
    void drawImGuiSuggestor();
    void drawImGuiTexture();
    void drawImPlotBarPlot();

    template <typename T> T getBufferValue(GLuint buffer, int index);
    void setOverlayTexel(int texelIndex, std::vector<float> color = {1.f, 1.f, 1.f, 1.f});
    void mirrorWeightsToOverlay();
    void brushTexels(uint centerTexelIndex, int radius, float weight);
    void eraseTexels(uint centerTexelIndex, int radius);
    void setTexelWeight(uint attributeIndex, float weight);
    float getTexelWeight(uint texelIndex);
    float getTexelValue(uint texelIndex, GLuint buffer);
    void autoweighAttributes(uint textureType);
    void autotagArchetypes();
    void clearAttributeWeights();
    void invertAttributeWeights();
    void refineAttributeWeights(uint textureType);
    void setClass(int i);
    void unsetClass(int i);
    void updateVisualizations(std::vector<uint> selectionCounts);
    void clearSelection();
    void clearSuggestions();
    void copyTextureDataToTextures();
    void addArchetype(uint archetypeDatapointIndex, uint archetypeClass);
    void removeArchetype(uint datapointIndex);
    void clearArchetypes();
    void updateSuggestions(uint nSuggestions = 0);
    void assess(uint symmetricSize);
    float sumWeightedAttributeValues(uint index);
    void updateDatapointIndicesForArchetypes();
    
  private:
    enum class BufferType {
      ePairwiseAttrDists,
      ePairsSelectedPerDatapoint,

      Length
    };

    enum class BufferTempType {
      eDatasetSelection,
      eIndicesSelection,
      eArchetypeTemp,

      Length
    };

    enum TabUpperType {
      eSelectionPrimary,
      eSelectionSecondary,
      eSelectionDifference,
      ePairwise,
      eSuggestions,
    };

    enum TabType {
      eAveragePrimary,
      eVariancePrimary,
      eAverageSecondary,
      eVarianceSecondary,
      eAverageDifference,
      eVarianceDifference,

      ePairwiseDiffsAll,
      ePairwiseDiffsInterclass,
      ePairwiseDiffsIntraclass,
      ePairwiseDiffsInterselection,
      ePairwiseDiffsIntraselection,

      eOverlay,

      Length
    };

    enum class ProgramType {
      ePairwiseAttrDiffsComp,
      dGuessClasses,

      Length
    };

    const std::array<const std::string, TabType::Length> _promptsValuetype = {
      "Mean: %0.2f",
      "Variance: %0.2f",
      "Mean: %0.2f",
      "Variance: %0.2f",
      "Difference in mean: %0.2f",
      "Difference in variance: %0.2f",
      "Pairwise difference (all): %0.2f",
      "Pairwise difference (interclass): %0.2f",
      "Pairwise difference (intraclass): %0.2f",
      "Pairwise difference (interselection): %0.2f",
      "Pairwise difference (intraselection): %0.2f",
      "Value: %0.2f"};

    const std::array<const std::string, TabType::Length> _promptsDenomtype = {
      "%u primary selected datapoints",
      "%u primary selected datapoints",
      "%u secondary selected datapoints",
      "%u secondary selected datapoints",
      "",
      "",
      "%u selected ",
      "%u selected interclass ",
      "%u selected intraclass ",
      "%u selected interselection ",
      "%u selected intraselection ",
      ""};
    
    std::array<const char*, 8> _buttons = {"A", "B", "C", "D", "E", "F", "G", "H"};

    std::array<ImVec4, 8> _buttonsColors = {ImVec4(1,0,0,1), ImVec4(0,1,0,1), ImVec4(0,0,1,1),
                                            ImVec4(1,1,0,1), ImVec4(1,0,1,1), ImVec4(0,1,1,1),
                                            ImVec4(1,1,1,1), ImVec4(0.5f,0.5f,0.5f,1)};

    // State
    bool _isInit;
    sne::Params* _params;
    dh::vis::Input _input;

    // ImGui state
    std::vector<uint> _selectionCounts;
    float _similarityWeight;
    float _attributeWeight;
    int _brushRadius;
    float _autoweighPercentage;
    float _autotagPercentage;
    uint _buttonPressed;
    int _tabIndex;
    uint _tabIndexPrev;
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
    uint _suggestionLevel;
    std::vector<uint> _denominators;
    std::vector<uint> _archetypeLabels; // Same order as _archetypeDatapointIndices
    std::vector<uint> _archetypeDatapointIndices; // Same order as _archetypeLabels
    std::unordered_map<uint, uint> _datapointArchetypeMapping;
    int _archetypeClassSelected;
    std::vector<uint> _suggestionDatapointIndices;
    uint _selectedDatapoint;
    bool _separationMode;

    // Objects
    sne::MinimizationBuffers _minimizationBuffers;
    sne::SimilaritiesBuffers _similaritiesBuffers;
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<BufferTempType, GLuint> _buffersTemp;
    util::EnumArray<TabType, GLuint> _buffersTextureData;
    util::EnumArray<TabType, GLuint> _textures;
    std::vector<GLuint> _suggestionTextures;
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
    std::vector<uint> getArchetypeLabels() { return _archetypeLabels; }
    std::vector<uint> getArchetypeIndices() { return _archetypeDatapointIndices; }
    std::unordered_map<uint, uint> getDatapointArchetypeMapping() { return _datapointArchetypeMapping; }
    void setDatapointArchetypeMapping(std::unordered_map<uint, uint> datapointArchetypeMapping) { _datapointArchetypeMapping = datapointArchetypeMapping; }
    void setArchetypeLabels(std::vector<uint> archetypeLabels) { _archetypeLabels = archetypeLabels; }
    void setArchetypeIndices(std::vector<uint> archetypeIndices) { _archetypeDatapointIndices = archetypeIndices; }
    uint getSelectedDatapoint() { return _selectedDatapoint; }
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
      swap(a._autoweighPercentage, b._autoweighPercentage);
      swap(a._autotagPercentage, b._autotagPercentage);
      swap(a._vizAllPairs, b._vizAllPairs);
      swap(a._denominators, b._denominators);
      swap(a._tabIndex, b._tabIndex);
      swap(a._classTextures, b._classTextures);
      swap(a._classCounts, b._classCounts);
      swap(a._classCountsSelected, b._classCountsSelected);
      swap(a._classNames, b._classNames);
      swap(a._classIsSet, b._classIsSet);
      swap(a._classesSet, b._classesSet);
      swap(a._setChanged, b._setChanged);
      swap(a._suggestionLevel, b._suggestionLevel);
      swap(a._classButtonPressed, b._classButtonPressed);
      swap(a._archetypeLabels, b._archetypeLabels);
      swap(a._archetypeDatapointIndices, b._archetypeDatapointIndices);
      swap(a._archetypeClassSelected, b._archetypeClassSelected);
      swap(a._suggestionDatapointIndices, b._suggestionDatapointIndices);
      swap(a._selectedDatapoint, b._selectedDatapoint);
      swap(a._separationMode, b._separationMode);
      swap(a._datapointArchetypeMapping, b._datapointArchetypeMapping);
      swap(a._buffers, b._buffers);
      swap(a._buffersTemp, b._buffersTemp);
      swap(a._buffersTextureData, b._buffersTextureData);
      swap(a._textures, b._textures);
      swap(a._classColors, b._classColors);
      swap(a._bufferClassColors, b._bufferClassColors);
      swap(a._minimizationBuffers, b._minimizationBuffers);
      swap(a._similaritiesBuffers, b._similaritiesBuffers);
      swap(a._programs, b._programs);
    }
  };
} // dh::vis