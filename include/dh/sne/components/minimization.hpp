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
#include "dh/vis/components/selection_render_task.hpp"
#include "dh/vis/components/embedding_render_task.hpp"
#include "dh/vis/components/border_render_task.hpp"

namespace dh::sne {
  template <uint D> // Dimension of produced embedding
  class Minimization {
    // aligned types
    using Bounds = util::AlignedBounds<D>;
    using vec = util::AlignedVec<D, float>;
    using uvec = util::AlignedVec<D, uint>;

  public:
    // Constr/destr
    Minimization();
    Minimization(Similarities* similarities, Params params);  
    ~Minimization(); 

    // Copy constr/assignment is explicitly deleted
    Minimization(const Minimization&) = delete;
    Minimization& operator=(const Minimization&) = delete;

    // Move constr/operator moves handles
    Minimization(Minimization&&) noexcept;
    Minimization& operator=(Minimization&&) noexcept;

    void initializeEmbeddingRandomly();

    // Computation
    void comp();                                            // Compute full minimization (i.e. params.iterations)
    void compIteration();                                   // Compute a single iteration: minimization + selection + translation
    void compIterationMinimizationRestart();                // Compute a restart of the minimization
    void compIterationMinimization();                       // Compute the minimization part of a single iteration
    void compIterationSelection();                          // Compute the selection part of a single iteration
    void compIterationTranslation();                        // Compute the translation part of a single iteration
    void checkBuffer(GLuint handle); ////

  private:
    enum class BufferType {
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
      eFixed,
      eTranslating,
      eEmbeddingRelative,
      eEmbeddingRelativeBeforeTranslation,

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
    Params _params;
    Similarities* _similarities;
    SimilaritiesBuffers _similaritiesBuffers;
    uint _iteration;
    Bounds _bounds;
    Bounds _boundsPrev;
    vis::Input _input;
    glm::vec2 _cursorPos; // Position of the mouse in embedding space, i.e. relative to _bounds
    glm::vec2 _cursorPosPrev;
    uint _cursorMode;
    uint _colorMapping;
    uint _colorMappingPrev;
    int _selectionRadius;
    int _selectionRadiusPrev;
    bool _mouseRightPrev;
    bool _reinit;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;
    std::vector<GLuint> _textures;
    GLuint _averageSelectionTexture;

    // Subcomponents
    Field<D> _field;
    std::shared_ptr<vis::SelectionInputTask> _selectionInputTask;
    std::shared_ptr<vis::SelectionRenderTask> _selectionRenderTask;
    std::shared_ptr<vis::EmbeddingRenderTask<D>> _embeddingRenderTask;
    std::shared_ptr<vis::BorderRenderTask<D>> _borderRenderTask;

  public:
    // Getters
    MinimizationBuffers buffers() const {
      return {
        _buffers(BufferType::eEmbedding),
        _buffers(BufferType::eField),
        _buffers(BufferType::eBounds),
        _buffers(BufferType::eSelected),
        _buffers(BufferType::eNeighborhoodPreservation),
        _averageSelectionTexture
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
      swap(a._iteration, b._iteration);
      swap(a._buffers, b._buffers);
      swap(a._textures, b._textures);
      swap(a._averageSelectionTexture, b._averageSelectionTexture);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._field, b._field);
      swap(a._selectionInputTask, b._selectionInputTask);
      swap(a._selectionRenderTask, b._selectionRenderTask);
      swap(a._embeddingRenderTask, b._embeddingRenderTask);
      swap(a._borderRenderTask, b._borderRenderTask);
    }
  };
} // dh::sne
