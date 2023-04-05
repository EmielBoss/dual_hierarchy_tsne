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

#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include <resource_embed/resource_embed.hpp>
#include "dh/constants.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/io.hpp"
#include "dh/util/debug.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/gl/buffertools.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/util/cu/knn.cuh"
#include <faiss/VectorTransform.h>
#include <imgui.h>
#include <implot.h>
#include <numeric> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Minimization]");

  // Params for field size
  constexpr uint fieldMinSize = 5;

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization()
  : _isInit(false) {
    // ...
  }

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, Params* params, std::vector<char> axisMapping)
  : _isInit(false), _loggedNewline(false), _similarities(similarities), _similaritiesBuffers(similarities->getBuffers()),
    _selectionCounts(2, 0), _params(params), _axisMapping(axisMapping), _axisMappingPrev(axisMapping), _axisIndexPrev(-1),
    _selectedDatapointPrev(0), _iteration(0), _iterationIntense(1000), _removeExaggerationIter(_params->nExaggerationIters) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eNeighborhoodPreservationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/neighborhood_preservation.comp"));
      _programs(ProgramType::eCountSelectedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/selection_count.comp"));
      if constexpr (D == 2) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/gradients.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/centerEmbedding.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
      }
      if constexpr (DD == 2) {
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selection.comp"));
      } else if constexpr (DD == 3) {
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/selection.comp"));
      }
      if constexpr (D == 2 && DD == 3) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D_3DD/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D_3DD/translation.comp"));
      } else if constexpr (D == 2) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/translation.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/translation.comp"));
      }
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      const std::vector<vec> zerovecs(_params->n, vec(0));
      const std::vector<vec> unitvecs(_params->n, vec(1));
      const std::vector<uint> falses(_params->n, 0); // TODO: use bools instead of uints (but I can't seem to initialize buffers with bools; std::vector specializes <bool>)
      const std::vector<float> ones(_params->n, 1.0f);
      std::vector<float> zeros(_params->n, 0.0f);
      std::vector<float> data;
      data.assign(dataPtr, dataPtr + _params->n * _params->nHighDims);
      if(_params->uniformDims || _params->imageDataset) { dh::util::normalizeData(data, _params->n, _params->nHighDims); }
      else { dh::util::normalizeDataNonUniformDims(data, _params->n, _params->nHighDims); }

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params->n * _params->nHighDims * sizeof(float), data.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eLabels), _params->n * sizeof(int), labelPtr, 0);
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params->n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelative), _params->n * sizeof(vecc), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelativeBeforeTranslation), _params->n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), unitvecs.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eBoundsSelection), 2 * sizeof(vec), unitvecs.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _params->n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), _params->n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), _params->n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), _params->n * sizeof(vec), zerovecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), _params->n * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborsEmb), _params->n * _params->k * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eDistancesEmb), _params->n * _params->k * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborhoodPreservation), _params->n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelection), _params->n * sizeof(uint), falses.data(), GL_DYNAMIC_STORAGE_BIT);
      // glNamedBufferStorage(_buffers(BufferType::eLabeled), _params->n * sizeof(uint), labeled.data(), 0); // Indicates whether datapoints are labeled
      dh::util::indicateLabeled(labelPtr, _params->n, _params->nClasses, 1, _buffers(BufferType::eLabeled));
      glNamedBufferStorage(_buffers(BufferType::eDisabled), _params->n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are disabled/inactive/"deleted"
      glNamedBufferStorage(_buffers(BufferType::eFixed), _params->n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eTranslating), _params->n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are being translated
      glNamedBufferStorage(_buffers(BufferType::eWeights), _params->n * sizeof(float), ones.data(), 0); // The attractive force multiplier per datapoint
      glAssert();
    }

    // Calculate nPCs number of principal components and stores the values in _pcs
    if(!_params->disablePCA) {
      faiss::PCAMatrix matrixPCA(_params->nHighDims, _params->nPCs);
      matrixPCA.train(_params->n, dataPtr);
      _pcs = matrixPCA.apply(_params->n, dataPtr);
    }

    initializeEmbeddingRandomly(_params->seed);

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Setup field subcomponent
    _field = Field<D>(buffers(), _params);

#ifdef DH_ENABLE_VIS_EMBEDDING
    // Setup input tasks
    if (auto& queue = vis::InputQueue::instance(); queue.isInit()) {
      _selectionInputTask = std::dynamic_pointer_cast<vis::SelectionInputTask>(vis::InputQueue::instance().find("SelectionInputTask"));
      if(DD == 3) { _trackballInputTask = std::dynamic_pointer_cast<vis::TrackballInputTask>(vis::InputQueue::instance().find("TrackballInputTask")); }
    }

    // Setup render tasks
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      std::string axistypesAbbr = "tpa-"; // Used to determine index of the selected axistype
      _axesRenderTask = queue.emplace(vis::AxesRenderTask<DD>(buffers(), _params, _axisMapping, axistypesAbbr.find(_axisMapping[2]), 1));
      _embeddingRenderTask = queue.emplace(vis::EmbeddingRenderTask<DD>(_params, 0, buffers()));
      _selectionRenderTask = queue.emplace(vis::SelectionRenderTask(_params, 5));
      _attributeRenderTask = queue.emplace(vis::AttributeRenderTask(_params, 10, buffers(), _similaritiesBuffers, _embeddingRenderTask->getColorBuffer(), labelPtr));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    _klDivergence = KLDivergence(_params, _similaritiesBuffers, buffers());

    _isInit = true;
    glAssert();
  }

  // Generate randomized embedding data
  // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
  template <uint D, uint DD>
  void Minimization<D, DD>::initializeEmbeddingRandomly(int seed) {
    
    // Copy over embedding buffer to host
    std::vector<vec> embedding(_params->n);
    glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params->n * sizeof(vec), embedding.data());

    // Seed the (bad) rng
    std::srand(seed);
    
    // Generate n random D-dimensional vectors
    for (uint i = 0; i < _params->n; ++i) {
      vec v;
      float r;

      do {
        r = 0.f;
        for (uint j = 0; j < D; ++j) {
          v[j] = 2.f * (static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.f)) - 1.f;
        }
        r = dot(v, v);
      } while (r > 1.f || r == 0.f);

      r = std::sqrt(-2.f * std::log(r) / r);
      embedding[i] = v * r * _params->rngRange;
    }

    // Copy to buffer
    glNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params->n * sizeof(vec), embedding.data());
    glAssert();
  }

  template <uint D, uint DD>
  Minimization<D, DD>::~Minimization() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _isInit = false;
    }
  }

  template <uint D, uint DD>
  Minimization<D, DD>::Minimization(Minimization<D, DD>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D, uint DD>
  Minimization<D, DD>& Minimization<D, DD>::operator=(Minimization<D, DD>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // Restarts the minimization
  template <uint D, uint DD>
  void Minimization<D, DD>::restartMinimization() {
    if(_iteration < 100) { return; }
    if(_input.alt) { initializeEmbeddingRandomly(_iteration); } else
    if(_input.num >= 0) { initializeEmbeddingRandomly(_input.num); }
    else { initializeEmbeddingRandomly(_params->seed); }
    _iteration = 0;
    _iterationIntense = 1000;
    restartExaggeration(_params->nExaggerationIters);
    const std::vector<vec> unitvecs(_params->n, vec(1));
    glClearNamedBufferData(_buffers(BufferType::ePrevGradients), GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glClearNamedBufferData(_buffers(BufferType::eGain), GL_R32F, GL_RED, GL_FLOAT, unitvecs.data());
  }

  // Restarts the exaggeration by pushing the exaggeration end iteration further ahead
  template <uint D, uint DD>
  void Minimization<D, DD>::restartExaggeration(uint nExaggerationIters) {
    _removeExaggerationIter = _iteration + nExaggerationIters;
  }

  // Configures the axes on request of change
  // template <uint D, uint DD>
  // void Minimization<D, DD>::reconfigureZAxis() {
  //   if constexpr (D == DD) { return; }

  //   std::vector<float> axisVals(_params->n);
  //   if(_axisMapping[2] == 'p') {
  //     for(uint i = 0; i < _params->n; ++i) { axisVals[i] = _pcs[i*_params->nPCs + _axisIndex]; }
  //   } else
  //   if(_axisMapping[2] == 'a') {
  //     for(uint i = 0; i < _params->n; ++i) { axisVals[i] = _dataPtr[i*_params->nHighDims + _axisIndex]; }
  //     if(_params->imageDataset) { _attributeRenderTask->setOverlayTexel(_axisIndex, {0.f, 1.f, 0.f, 1.f}); }
  //   }
  //   auto [minIt, maxIt] = std::minmax_element(axisVals.begin(), axisVals.end());
  //   float min = *minIt;
  //   float range = *maxIt - *minIt;
  //   float rangeInv = range > 0 ? 1 / range : 1;
  //   uint stride = (DD == 2) ? 2 : 4;
  //   for(uint i = 0; i < _params->n; ++i) {
  //     float valRel = (axisVals[i] - min) * rangeInv;
  //     glNamedBufferSubData(_buffers(BufferType::eEmbeddingRelative), (i*stride + D) * sizeof(float), sizeof(float), &valRel);
  //   }
  //   glAssert();
  // }

  template <uint D, uint DD>
  void Minimization<D, DD>::deselect() {
    std::fill(_selectionCounts.begin(), _selectionCounts.end(), 0);
    _selectionRenderTask->setSelectionCounts(_selectionCounts);
    _embeddingRenderTask->setWeighForces(true); // Use force weighting again; optional but may be convenient for the user
    _attributeRenderTask->clear();
    glClearNamedBufferData(_buffers(BufferType::eSelection), GL_R32I, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::selectAll() {
    dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 1, 0, _buffers(BufferType::eDisabled));
    if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
    compIterationSelect(true);
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::selectInverse() {
    dh::util::BufferTools::instance().flip<uint>(_buffers(BufferType::eSelection), _params->n);
    if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
    compIterationSelect(true);
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::comp() {
    while (_iteration < _params->iterations) {
      compIteration();
    }
  }

  // Core function handling everything that needs to happen each frame
  template <uint D, uint DD>
  bool Minimization<D, DD>::compIteration() {
    _input = _selectionInputTask->getInput();

    _selectionRenderTask->setInput(_input);
    _attributeRenderTask->setInput(_input);

    // Send selection radius to selection render task
    _selectionRadiusRel = _input.mouseScroll / 100.f;
    _selectionRenderTask->setSelectionRadiusRel(_selectionRadiusRel);

    _selectionRenderTask->setMousePosScreen(_input.mousePosScreen); // Send screen position to GUI

    // Synchronize selection mode
    _selectOnlyLabeled = _selectionRenderTask->getSelectionMode();
    _embeddingRenderTask->setSelectionMode(_selectOnlyLabeled);

    // Get everything related with the cursor and selection brush
    if(_iteration < 1) { _window = util::GLWindow::currentWindow(); }
    glm::vec2 resolution = glm::vec2(_window->size());
    _model_view_2D = glm::translate(glm::vec3(-0.5f, -0.5f, -1.0f)); // TODO: get this directly from Rendered
    _proj_2D = glm::infinitePerspective(1.0f, resolution.x / resolution.y, 0.0001f); // TODO: get this directly from renderer
    if(DD == 3) {
      _model_view_3D = _trackballInputTask->matrix() * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f)); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)
      _proj_3D = glm::perspectiveFov(0.5f, resolution.x, resolution.y, 0.0001f, 1000.f); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)  
    }

    // Deselect
    if(_input.d) { deselect(); }
    
    // Clear translations if not translating
    if(!_input.mouseRight) { glClearNamedBufferData(_buffers(BufferType::eTranslating), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); }
    
    // Free/unfix fixed datapoints
    if(_input.f) {
      const std::vector<float> ones(_params->n, 1.0f);
      glClearNamedBufferData(_buffers(BufferType::eWeights), GL_R32F, GL_RED, GL_FLOAT, ones.data());
      glClearNamedBufferData(_buffers(BufferType::eFixed), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    }

    if(_input.r) { restartMinimization(); } // Restart
    if(_input.e) { restartExaggeration(3); } // Re-exaggerate

    _embeddingRenderTask->setIteration(_iteration);
    if(_iteration % _iterationIntense == 0 && _iterationIntense > 10) { _iterationIntense /= 2; }
    if(_iteration % _iterationIntense == 0 || _embeddingRenderTask->getButtonPressed() == 2) { _embeddingRenderTask->setKLDivergence(_klDivergence.comp()); }

    if(!_input.space) { compIterationMinimize(); } // Compute iteration, or pause if space is pressed
    if(_input.mouseLeft || _input.mouseMiddle || _input.u) { compIterationSelect(); } // Select
    if(_input.mouseRight || _mouseRightPrev) { compIterationTranslate(); } // Translate
    
    if(_input.del) { // Disable
      uint nEnabled = dh::util::BufferTools::instance().reduce<uint>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      float fracDisabled = (float) _selectionCounts[0] / (float) nEnabled;
      dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eDisabled), _params->n, 1, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 0, 1, _buffers(BufferType::eDisabled));
      compIterationSelect(true);
      _similarities->weighSimilarities(1.f / (1.f - fracDisabled));
      _embeddingRenderTask->setPointRadius(std::min(100.f / (nEnabled - _selectionCounts[0]), 0.005f));
    }
    if(_input.ins) { // Enable
      uint nEnabled = dh::util::BufferTools::instance().reduce<uint>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      float fracEnabled = (float) nEnabled / (float) _params->n;
      _similarities->weighSimilarities(fracEnabled);
      glClearNamedBufferData(_buffers(BufferType::eDisabled), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      _embeddingRenderTask->setPointRadius(100.f / _params->n);
    }

    _mousePosClipPrev = _input.mousePosClip;
    _mouseRightPrev = _input.mouseRight;

    // Process attribute selection texture buttons
    int button = _selectionRenderTask->getButtonPressed();
    if(button > 0 && button != _buttonSelectionPrev) {
      if(button == 1) { // Apply similarity weight
        _similarities->weighSimilarities(_selectionRenderTask->getSimilarityWeight(), _buffers(BufferType::eSelection));
        _similarities->renormalizeSimilarities();
      }
      if(button == 10) { // Apply similarity weight to intersimilarities between selections
        _similarities->weighSimilarities(_selectionRenderTask->getSimilarityWeight(), _buffers(BufferType::eSelection), true);
        _similarities->renormalizeSimilarities(_buffers(BufferType::eSelection));
      }
      if(button == 20) { // Select all
        selectAll();
      }
      if(button == 30) { // Select inverse
        selectInverse();
      }
    }
    _buttonSelectionPrev = button;

    button = _attributeRenderTask->getButtonPressed();
    if(button > 0 && button != _buttonAttributePrev) {
      std::set<uint> weightedAttributeIndices = _attributeRenderTask->getWeightedAttributeIndices();
      if(button == 2) { // Recalc similarities (ratio)
        _similarities->weighSimilaritiesPerAttributeRatio(weightedAttributeIndices, _buffers(BufferType::eSelection), _selectionCounts[0], _buffers(BufferType::eLabels));
      }
      if(button == 25) { // Recalc similarities (range)
        _similarities->weighSimilaritiesPerAttributeRange(weightedAttributeIndices, _buffers(BufferType::eSelection), _selectionCounts[0], _buffers(BufferType::eLabels));
      }
      if(button == 26) { // Recalc similarities (resemble)
        _similarities->weighSimilaritiesPerAttributeResemble(weightedAttributeIndices, _buffers(BufferType::eSelection), _selectionCounts[0], _buffers(BufferType::eLabels), _attributeRenderTask->getSnapslotHandles(), _params->nHighDims);
      }
      if(button == 3) { // Reset similarities
        _similarities->reset();
      }
    }
    _buttonAttributePrev = button;

    if(_embeddingRenderTask->getButtonPressed() == 1) {
      uint n = _params->n;
      _similarities->recomp(_buffers(BufferType::eSelection), _embeddingRenderTask->getPerplexity(), _embeddingRenderTask->getK());
      _similaritiesBuffers = _similarities->getBuffers(); // Refresh buffer handles, because recomp() deletes and recreates buffers
      dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eDataset), n, _params->nHighDims, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eEmbeddingRelative), n, D, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eWeights), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<uint>(_buffers(BufferType::eLabels), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<uint>(_buffers(BufferType::eLabeled), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<uint>(_buffers(BufferType::eFixed), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<uint>(_buffers(BufferType::eDisabled), n, 1, _buffers(BufferType::eSelection));
      deselect();
      uint nEnabled = dh::util::BufferTools::instance().reduce<uint>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      _embeddingRenderTask->setPointRadius(std::min(100.f / (nEnabled - _selectionCounts[0]), 0.005f));
      _embeddingRenderTask->setMinimizationBuffers(buffers()); // Update buffer handles, because BufferTools::remove() deletes and recreates buffers
      _attributeRenderTask->setMinimizationBuffers(buffers());
      _attributeRenderTask->setSimilaritiesBuffers(_similaritiesBuffers);
      restartMinimization();
    }

    int classToSelect = _attributeRenderTask->getClassButtonPressed();
    if(classToSelect >= 0) {
      dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 1, (uint) classToSelect, _buffers(BufferType::eLabels));
      if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<uint>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
      compIterationSelect(true);
    }

    // Select individual datapoints if int field input changed
    uint selectedDatapoint = (uint) _selectionRenderTask->getSelectedDatapoint();
    if(selectedDatapoint != _selectedDatapointPrev && selectedDatapoint < _params->n) {
      uint sn = _input.s + 1; // Selection number
      glNamedBufferSubData(_buffers(BufferType::eSelection), selectedDatapoint * sizeof(uint), sizeof(uint), &sn);
      compIterationSelect(true);
      _selectedDatapointPrev = selectedDatapoint;
    }

    if(_embeddingRenderTask->getButtonPressed() == 10) { // Import state
      dh::util::readState(_params->n, _params->nHighDims, D, _buffers, _similaritiesBuffers.attributeWeights, _attributeRenderTask->getWeightedAttributeIndices(), _attributeRenderTask->getSnapslotHandles());
      _embeddingRenderTask->setMinimizationBuffers(buffers());
      _attributeRenderTask->setMinimizationBuffers(buffers());
      _attributeRenderTask->mirrorWeightsToOverlay();
      compIterationSelect(true);
    } else
    if(_embeddingRenderTask->getButtonPressed() == 11) { // Export state
      dh::util::writeState(_params->n, _params->nHighDims, D, _buffers, _similaritiesBuffers.attributeWeights, _attributeRenderTask->getWeightedAttributeIndices(), _attributeRenderTask->getSnapslotHandles());
    }

    // Reset some stuff upon axis change
    // _axisMapping = _axesRenderTask->getAxisMapping();
    // _axisIndex = _axesRenderTask->getSelectedIndex();
    // if(_axisIndex != _axisIndexPrev || _axisMapping != _axisMappingPrev) {
    //   reconfigureZAxis();
    //   _axisMappingPrev = _axisMapping;
    //   _axisIndexPrev = _axisIndex;
    // }

    // Reconstruct this Minimization if adding subtracting a t-SNE dimension
    // _axisMapping = _axesRenderTask->getAxisMapping();
    // _axisIndex = _axesRenderTask->getSelectedIndex();
    // if((_axisMapping[2] == 't' || _axisMappingPrev[2] == 't') && _axisMapping[2] != _axisMappingPrev[2]) {
    //   return true;
    // }

    glAssert();
    return false;
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationMinimize() {

    // 1.
    // Compute embedding bounds
    {
      auto& timer = _timers(TimerType::eBoundsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eBoundsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eTranslating));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eBounds));

      // Dispatch shader
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Copy bounds back to host (hey look: an expensive thing I shouldn't be doing)
    _boundsPrev = _bounds;
    glGetNamedBufferSubData(_buffers(BufferType::eBounds), 0, sizeof(Bounds),  &_bounds);

    // 2.
    // Perform field approximation in subcomponent
    {
      // Determine field texture size by scaling bounds
      const vec range = _bounds.range();
      const float ratio = (D == 2) ? _params->fieldScaling2D : _params->fieldScaling3D;
      uvec size = dh::util::max(uvec(range * ratio), uvec(fieldMinSize));

      // Size becomes nearest larger power of two for field hierarchy
      size = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(size.x)) / glm::log(2.f))));

      // Delegate to subclass
      _field.comp(size, _iteration);
    }

    // 3.
    // Compute Z, ergo a reduction over q_{ij}
    {
      auto& timer = _timers(TimerType::eZComp);
      timer.tick();

      auto& program = _programs(ProgramType::eZComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDisabled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eZ));

      // Dispatch shader
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 4.
    // Compute attractive forces
    { 
      auto& timer = _timers(TimerType::eAttractiveComp);
      timer.tick();

      auto& program = _programs(ProgramType::eAttractiveComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPos", _params->n);
      program.template uniform<float>("invPos", 1.f / static_cast<float>(_params->n));
      program.template uniform<float>("weightFalloff", _embeddingRenderTask->getWeightFalloff());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDisabled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.layout);  // n structs of two uints; the first is the offset into _similaritiesBuffers.neighbors where its kNN set starts, the second is the size of its kNN set
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.neighbors); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _similaritiesBuffers.similarities); // Corresponding similarities
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u / 32u), 1, 1); // One warp/subgroup per datapoint
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Compute exaggeration factor
    float exaggeration = 1.0f;
    if (_iteration <= _removeExaggerationIter) {
      exaggeration = _params->exaggerationFactor;
    } else if (_iteration <= _removeExaggerationIter + _params->nExponentialDecayIters) {
      float decay = 1.0f - static_cast<float>(_iteration - _removeExaggerationIter)
                         / static_cast<float>(_params->nExponentialDecayIters);
      exaggeration = 1.0f + (_params->exaggerationFactor - 1.0f) * decay;
    }

    // 5.
    // Compute gradients
    {
      auto& timer = _timers(TimerType::eGradientsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eGradientsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttractive));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Precompute instead of doing it in shader N times
    const float iterMult = (static_cast<double>(_iteration) < _params->momentumSwitchIter) 
                         ? _params->momentum 
                         : _params->finalMomentum;

    // 6.
    // Update embedding
    {
      auto& timer = _timers(TimerType::eUpdateEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eUpdateEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<float>("eta", _params->eta);
      program.template uniform<float>("minGain", _params->minimumGain);
      program.template uniform<float>("mult", 1.0);
      program.template uniform<float>("iterMult", iterMult);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eGain));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 7.
    // Re-center embedding
    {
      const vec boundsCenter = _bounds.center();
      const vec boundsRange = _bounds.range();
      float scaling = 1.0f;
      if (exaggeration > 1.2f && boundsRange.y < 0.1f) {
        scaling = 0.1f / boundsRange.y;
      }
      
      auto& timer = _timers(TimerType::eCenterEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eCenterEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<float>("scaling", scaling);
      program.template uniform<float, D>("center", boundsCenter);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 8.
    // Compute neighborhood preservation per datapoint
    _colorMapping = _embeddingRenderTask->getColorMapping();
    if(_colorMapping == 2 && _colorMappingPrev != 2) {
      // Compute approximate KNN of each point in embedding, delegated to FAISS
      std::vector<vec> embedding(_params->n);
      glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params->n * sizeof(vec), embedding.data());
      std::vector<float> embeddingFiltered = dh::util::to_unaligned_vector<D, float>(embedding, _params->n);
      {
        util::KNN knn(
          embeddingFiltered.data(),
          _buffers(BufferType::eDistancesEmb),
          _buffers(BufferType::eNeighborsEmb),
          _params->n, _params->k, D);
        knn.comp();
      }

      {
        auto& program = _programs(ProgramType::eNeighborhoodPreservationComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params->n);
        program.template uniform<uint>("kParam", _params->k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _similaritiesBuffers.layout);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similaritiesBuffers.neighbors);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighborsEmb));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighborhoodPreservation));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        glAssert();
      }
      _colorMappingPrev = _colorMapping;
    }

    // Log progress; spawn progressbar on the current (new on first iter) line
    // reporting current iteration and size of field texture
    if (!_loggedNewline) {
      _loggedNewline = true;
    }
    if ((++_iteration % 100) == 0) {
      // Assemble string to print field's dimensions and memory usage
      std::stringstream fieldStr;
      {
        const uvec fieldSize = _field.size();
        fieldStr << fieldSize[0] << "x" << fieldSize[1];
        if constexpr (D == 3) {
          fieldStr << "x" << fieldSize[2];
        }
        fieldStr << " (" << (static_cast<float>(_field.memSize()) / 1'048'576.0f) << " mb)";
      }
      
      const std::string postfix = (_iteration < _params->iterations)
                                ? "iter: " + std::to_string(_iteration) + ", field: " + fieldStr.str()
                                : "Done!";
      util::ProgressBar progressBar(prefix + "Computing...", postfix);
      progressBar.setProgress(static_cast<float>(_iteration) / static_cast<float>(_params->iterations));
    }
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationSelect(bool skipEval) {
    uint si = _input.s; // Selection index
    uint sn = si + 1; // Selection number

    // 1.
    // Compute selection (or select all)
    if(!skipEval) {
      auto& program = _programs(ProgramType::eSelectionComp);
      program.bind();

      // Set uniform
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<uint>("selectionNumber", sn);
      program.template uniform<float, 2>("mousePosClip", _input.mousePosClip);
      program.template uniform<float>("selectionRadiusRel", _selectionRadiusRel);
      program.template uniform<float>("selectOnlyLabeled", _selectOnlyLabeled);
      program.template uniform<float, 4, 4>("model_view", DD == 2 ? _model_view_2D : _model_view_3D);
      program.template uniform<float, 4, 4>("proj", DD == 2 ? _proj_2D : _proj_3D);
      program.template uniform<bool>("deselect", _input.mouseMiddle);
      program.template uniform<bool>("unfix", _input.u);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLabeled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDisabled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSelection));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eWeights));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    // 2.
    // Count number of selected datapoints
    for (uint s = 0; s < 2; ++s) {
      uint selectionCountPrev = _selectionCounts[0];
      _selectionCounts[s] = dh::util::BufferTools::instance().reduce<uint>(_buffers(BufferType::eSelection), 3, _params->n, 0, s + 1);

      _selectionRenderTask->setSelectionCounts(_selectionCounts);

      // Turn off force weighing if too many datapoints are selected at once, which is likely not what the user wants
      if(_selectionCounts[0] - selectionCountPrev > _params->n / 500) { _embeddingRenderTask->setWeighForces(false); }
    }

    _attributeRenderTask->update(_selectionCounts);
  }

  template <uint D, uint DD>
  void Minimization<D, DD>::compIterationTranslate() {

    if(!_mouseRightPrev) {
      glm::vec2 min = dh::util::BufferTools::instance().reduce<glm::vec2>(_buffers(BufferType::eEmbeddingRelative), 1, _params->n, _buffers(BufferType::eSelection));
      glm::vec2 max = dh::util::BufferTools::instance().reduce<glm::vec2>(_buffers(BufferType::eEmbeddingRelative), 2, _params->n, _buffers(BufferType::eSelection));
      glNamedBufferSubData(_buffers(BufferType::eBoundsSelection), 0 * sizeof(vec), sizeof(vec), &min);
      glNamedBufferSubData(_buffers(BufferType::eBoundsSelection), 1 * sizeof(vec), sizeof(vec), &max);
    }

    glm::vec4 mousePosRel = glm::inverse(_proj_2D * _model_view_2D) * glm::vec4(_input.mousePosClip, 0.9998, 1);
    glm::vec4 mousePosRelPrev = glm::inverse(_proj_2D * _model_view_2D) * glm::vec4(_mousePosClipPrev, 0.9998, 1);
    vec shiftRel = mousePosRel - mousePosRelPrev;
    if constexpr (D == 3) {
      shiftRel = _trackballInputTask->changeOfBaseToCam() * shiftRel;
      shiftRel.y = -shiftRel.y;
    }

    auto& program = _programs(ProgramType::eTranslationComp);
    program.bind();

    // Set uniform
    program.template uniform<uint>("nPoints", _params->n);
    program.template uniform<float, D>("shiftRel", shiftRel);
    program.template uniform<bool>("translationStarted", !_mouseRightPrev);
    program.template uniform<bool>("translationFinished", !_input.mouseRight && _mouseRightPrev);
    program.template uniform<bool>("weighForces", _embeddingRenderTask->getWeighForces());
    program.template uniform<float>("weight", _embeddingRenderTask->getWeightFixed());

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelection));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbedding));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingRelative));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eFixed));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eTranslating));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eWeights));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBounds));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eBoundsSelection));

    // Dispatch shader
    glDispatchCompute(ceilDiv(_params->n, 256u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  // Template instantiations for 2/3 dimensions
  template class Minimization<2, 2>;
  template class Minimization<2, 3>;
  template class Minimization<3, 3>;
} // dh::sne