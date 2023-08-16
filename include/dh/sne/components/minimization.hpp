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
#include "dh/sne/components/kl_divergence.hpp"
#include "dh/vis/render_queue.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/vis/components/selection_input_task.hpp"
#include "dh/vis/components/trackball_input_task.hpp"
#include "dh/vis/components/selection_render_task.hpp"
#include "dh/vis/components/embedding_render_task.hpp"
#include "dh/vis/components/axes_render_task.hpp"
#include "dh/vis/components/attribute_render_task.hpp"

namespace dh::sne {
  template <uint D>
  class Minimization {
    // aligned types
    using Bounds = util::AlignedBounds<D>;
    using vec = util::AlignedVec<D, float>;
    using uvec = util::AlignedVec<D, uint>;

  public:
    // Constr/destr
    Minimization();
    Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, const float* colorPtr, Params* params);
    ~Minimization();

    // Copy constr/assignment is explicitly deleted
    Minimization(const Minimization&) = delete;
    Minimization& operator=(const Minimization&) = delete;

    // Move constr/operator moves handles
    Minimization(Minimization&&) noexcept;
    Minimization& operator=(Minimization&&) noexcept;

    void initializeEmbeddingRandomly(int seed);
    void deselect();
    void selectAll();
    void selectInverse();
    void restartMinimization();
    void restartExaggeration(uint nExaggerationIters);
    void syncBufferHandles();
    void stateImport();
    void stateExport();

    // Computation
    void comp();                                                                // Compute full minimization (i.e. params.iterations)
    void compIteration();                                                       // Compute a single iteration: minimization + selection + translation
    void compIterationMinimize();                                               // Compute the minimization part of a single iteration
    void compIterationSelect(bool skipEval = false);                            // Compute the selection part of a single iteration
    void compIterationTranslate();                                              // Compute the translation part of a single iteration

  private:
    enum class BufferType {
      eLabels,
      eEmbedding,
      eBounds,
      eBoundsReduce,
      eBoundsSelection,
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
      eSelection,
      eFixed,
      eTranslating,
      eWeights,
      eLabeled,
      eEmbeddingRelative,
      eEmbeddingRelativeBeforeTranslation,
      eDisabled,

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
    Params* _params;
    Similarities* _similarities;
    SimilaritiesBuffers _similaritiesBuffers;
    util::GLWindow* _window;
    uint _iteration;
    uint _iterationIntense;
    uint _removeExaggerationIter;
    Bounds _bounds;
    Bounds _boundsPrev;
    vis::Input _input;
    uint _colorMapping;
    uint _colorMappingPrev;
    bool _selectOnlyLabeled;
    std::vector<uint> _selectionCounts;
    float _selectionRadiusRel;
    uint _selectedDatapointPrev;
    bool _mouseRightPrev;
    glm::vec2 _mousePosClipPrev;
    glm::mat4 _model_view_2D;
    glm::mat4 _proj_2D;
    glm::mat4 _model_view_3D;
    glm::mat4 _proj_3D;
    uint _buttonSelectionPrev;
    uint _buttonAttributePrev;
    bool __assessed;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    Field<D> _field;
    std::shared_ptr<vis::SelectionInputTask> _selectionInputTask;
    std::shared_ptr<vis::TrackballInputTask> _trackballInputTask;
    std::shared_ptr<vis::SelectionRenderTask> _selectionRenderTask;
    std::shared_ptr<vis::EmbeddingRenderTask<D>> _embeddingRenderTask;
    std::shared_ptr<vis::AxesRenderTask<D>> _axesRenderTask;
    std::shared_ptr<vis::AttributeRenderTask> _attributeRenderTask;
    KLDivergence _klDivergence;

  public:
    // Getters
    MinimizationBuffers buffers() const {
      return {
        _buffers(BufferType::eEmbedding),
        _buffers(BufferType::eEmbeddingRelative),
        _buffers(BufferType::eField),
        _buffers(BufferType::eBounds),
        _buffers(BufferType::eLabels),
        _buffers(BufferType::eLabeled),
        _buffers(BufferType::eSelection),
        _buffers(BufferType::eFixed),
        _buffers(BufferType::eDisabled),
        _buffers(BufferType::eNeighborhoodPreservation),
      };
    }
    bool isInit() const { return _isInit; }

    // std::swap impl
    friend void swap(Minimization<D>& a, Minimization<D>& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._similarities, b._similarities);
      swap(a._similaritiesBuffers, b._similaritiesBuffers);
      swap(a._selectedDatapointPrev, b._selectedDatapointPrev);
      swap(a._selectionCounts, b._selectionCounts);
      swap(a._iteration, b._iteration);
      swap(a._iterationIntense, b._iterationIntense);
      swap(a._removeExaggerationIter, b._removeExaggerationIter);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._field, b._field);
      swap(a._selectionInputTask, b._selectionInputTask);
      swap(a._trackballInputTask, b._trackballInputTask);
      swap(a._selectionRenderTask, b._selectionRenderTask);
      swap(a._embeddingRenderTask, b._embeddingRenderTask);
      swap(a._axesRenderTask, b._axesRenderTask);
      swap(a._attributeRenderTask, b._attributeRenderTask);
      swap(a._klDivergence, b._klDivergence);
      swap(a.__assessed, b.__assessed);
    }
  };
} // dh::sne