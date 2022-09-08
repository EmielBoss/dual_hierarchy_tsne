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

#include <memory>
#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/timer.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/similarities.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/sne/components/field.hpp"
#include "dh/vis/render_queue.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/vis/components/selection_input_task.hpp"
#include "dh/vis/components/trackball_input_task.hpp"
#include "dh/vis/components/selection_render_task.hpp"
#include "dh/vis/components/embedding_render_task.hpp"
#include "dh/vis/components/axes_render_task.hpp"

namespace dh::sne {
  template <uint D, uint DD> // Number of t-SNE axes and total number of axes
  class Minimization {
    // aligned types
    using Bounds = util::AlignedBounds<D>;
    using vec = util::AlignedVec<D, float>;
    using vecc = util::AlignedVec<DD, float>;
    using uvec = util::AlignedVec<D, uint>;

  public:
    // Constr/destr
    Minimization();
    Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, Params params, std::vector<char> axisMapping);
    ~Minimization();

    // Copy constr/assignment is explicitly deleted
    Minimization(const Minimization&) = delete;
    Minimization& operator=(const Minimization&) = delete;

    // Move constr/operator moves handles
    Minimization(Minimization&&) noexcept;
    Minimization& operator=(Minimization&&) noexcept;

    std::vector<float> normalizeDataset(const float* dataPtr);
    std::vector<float> normalizeDatasetUniformScale(const float* dataPtr);
    void initializeEmbeddingRandomly(int seed);
    void restartMinimization();
    void reconfigureZAxis();
    void setTexelValue(int texelIndex, int component, float texelVal = 1.f);
    void fillTextureComponent(uint component, float value, GLuint buffer = 0);
    std::vector<char> getAxisMapping() { return _axisMapping; }

    // Computation
    void comp();                                            // Compute full minimization (i.e. params.iterations)
    bool compIteration();                                   // Compute a single iteration: minimization + selection + translation
    void compIterationMinimization();                       // Compute the minimization part of a single iteration
    void compIterationSelection();                          // Compute the selection part of a single iteration
    void compIterationTranslation();                        // Compute the translation part of a single iteration
    template <typename T> void writeBuffer(GLuint handle, uint n, uint d, std::string filename); ////

  private:
    enum class BufferType {
      eDataset,
      eLabels,
      eEmbedding,
      eBounds,
      eBoundsReduce,
      eZ,
      eZReduce,
      eField,
      eAttractive,
      eGradients,
      ePrevGradients,
      eGain,
      eDistancesEmb,
      eNeighborsEmb,
      eNeighborhoodPreservation,
      eSelected,
      eSelectionCount,
      eSelectionCountReduce,
      eTextureDataReduce,
      eFixed,
      eTranslating,
      eWeights,
      eLabeled,
      eEmbeddingRelative,
      eEmbeddingRelativeBeforeTranslation,
      eTest,

      Length
    };

    enum class TextureType {
      eAverage,
      eVariance,

      Length
    };

    enum class ProgramType {
      eBoundsComp,
      eZComp,
      eAttractiveComp,
      eGradientsComp,
      eUpdateEmbeddingComp,
      eCenterEmbeddingComp,
      eNeighborhoodPreservationComp,
      eSelectionComp,
      eCountSelectedComp,
      eSelectionAverageComp,
      eSelectionVarianceComp,
      eTranslationComp,

      Length
    };

    enum class TimerType {
      eBoundsComp,
      eZComp,
      eAttractiveComp,
      eGradientsComp,
      eUpdateEmbeddingComp,
      eCenterEmbeddingComp,

      Length
    };

    // State
    bool _isInit;
    bool _loggedNewline;
    Params _params;
    std::vector<char> _axisMapping;
    std::vector<char> _axisMappingPrev;
    int _axisIndex;
    int _axisIndexPrev;
    Similarities* _similarities;
    SimilaritiesBuffers _similaritiesBuffers;
    const float* _dataPtr;
    float* _pcs;
    uint _iteration;
    Bounds _bounds;
    Bounds _boundsPrev;
    vis::Input _input;
    uint _cursorMode;
    uint _colorMapping;
    uint _colorMappingPrev;
    bool _selectOnlyLabeled;
    bool _selectOnlyLabeledPrev;
    uint _selectionCount;
    float _selectionRadiusRel;
    bool _mouseLeftPrev;
    bool _mouseRightPrev;
    glm::vec2 _mousePosClipPrev;
    glm::mat4 _model_view_2D;
    glm::mat4 _proj_2D;
    glm::mat4 _model_view_3D;
    glm::mat4 _proj_3D;
    int _draggedAttribute;
    int _draggedAttributePrev;
    uint _button;
    uint _buttonPrev;
    std::set<uint> _selectedAttributeIndices;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<TextureType, GLuint> _buffersTextureData;
    util::EnumArray<TextureType, GLuint> _textures;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    Field<D> _field;
    std::shared_ptr<vis::SelectionInputTask> _selectionInputTask;
    std::shared_ptr<vis::TrackballInputTask> _trackballInputTask;
    std::shared_ptr<vis::SelectionRenderTask> _selectionRenderTask;
    std::shared_ptr<vis::EmbeddingRenderTask<DD>> _embeddingRenderTask;
    std::shared_ptr<vis::AxesRenderTask<DD>> _axesRenderTask;

  public:
    // Getters
    MinimizationBuffers buffers() const {
      return {
        _buffers(BufferType::eEmbedding),
        _buffers(BufferType::eEmbeddingRelative),
        _buffers(BufferType::eField),
        _buffers(BufferType::eBounds),
        _buffers(BufferType::eLabeled),
        _buffers(BufferType::eSelected),
        _buffers(BufferType::eFixed),
        _buffers(BufferType::eNeighborhoodPreservation),
        _textures(TextureType::eAverage),
        _textures(TextureType::eVariance)
      };
    }
    bool isInit() const { return _isInit; }

    // std::swap impl
    friend void swap(Minimization<D, DD>& a, Minimization<D, DD>& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._axisMapping, b._axisMapping);
      swap(a._axisMappingPrev, b._axisMappingPrev);
      swap(a._axisIndex, b._axisIndex);
      swap(a._axisIndexPrev, b._axisIndexPrev);
      swap(a._similarities, b._similarities);
      swap(a._similaritiesBuffers, b._similaritiesBuffers);
      swap(a._dataPtr, b._dataPtr);
      swap(a._pcs, b._pcs);
      swap(a._draggedAttribute, b._draggedAttribute);
      swap(a._draggedAttributePrev, b._draggedAttributePrev);
      swap(a._selectionCount, b._selectionCount);
      swap(a._iteration, b._iteration);
      swap(a._buffers, b._buffers);
      swap(a._buffersTextureData, b._buffersTextureData);
      swap(a._textures, b._textures);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._field, b._field);
      swap(a._selectionInputTask, b._selectionInputTask);
      swap(a._trackballInputTask, b._trackballInputTask);
      swap(a._selectionRenderTask, b._selectionRenderTask);
      swap(a._embeddingRenderTask, b._embeddingRenderTask);
      swap(a._axesRenderTask, b._axesRenderTask);
    }
  };
} // dh::sne
