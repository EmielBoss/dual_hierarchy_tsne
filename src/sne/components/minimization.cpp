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
#include <imgui.h>
#include <implot.h>
#include <numeric> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Minimization]");

  // Params for field size
  constexpr uint fieldMinSize = 5;

  template <uint D>
  Minimization<D>::Minimization()
  : _isInit(false) {
    // ...
  }

  template <uint D>
  Minimization<D>::Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, const float* colorPtr, Params* params)
  : _isInit(false), _loggedNewline(false), _similarities(similarities), _similaritiesBuffers(similarities->getBuffers()), _separationMode(false),
    _selectionCounts(2, 0), _selectedDatapointPrevSRT(0), _selectedDatapointPrevART(-1), _params(params),
    _iteration(0), _iterationIntense(1000), _removeExaggerationIter(_params->nExaggerationIters), __assessed(false) {
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
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selection.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/translation.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/selection.comp"));
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
      const std::vector<int> falses(_params->n, 0); // TODO: use bools instead of ints (but I can't seem to initialize buffers with bools; std::vector specializes <bool>)
      const std::vector<float> ones(_params->n, 1.0f);

      glCreateBuffers(_buffers.size(), _buffers.data());
      // glNamedBufferStorage(_buffers(BufferType::eLabels), _params->n * sizeof(int), labelPtr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLabels), _params->n * sizeof(int), labelPtr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params->n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelative), _params->n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
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
      glNamedBufferStorage(_buffers(BufferType::eSelection), _params->n * sizeof(int), falses.data(), GL_DYNAMIC_STORAGE_BIT);
      // glNamedBufferStorage(_buffers(BufferType::eLabeled), _params->n * sizeof(int), labeled.data(), 0); // Indicates whether datapoints are labeled
      dh::util::indicateLabeled(labelPtr, _params->n, _params->nClasses, 1, _buffers(BufferType::eLabeled));
      glNamedBufferStorage(_buffers(BufferType::eDisabled), _params->n * sizeof(int), falses.data(), 0); // Indicates whether datapoints are disabled/inactive/"deleted"
      glNamedBufferStorage(_buffers(BufferType::eFixed), _params->n * sizeof(int), falses.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eTranslating), _params->n * sizeof(int), falses.data(), 0); // Indicates whether datapoints are being translated
      glNamedBufferStorage(_buffers(BufferType::eWeights), _params->n * sizeof(float), ones.data(), 0); // The attractive force multiplier per datapoint
      glNamedBufferStorage(_buffers(BufferType::eArchetypes), _params->n * sizeof(int), falses.data(), GL_DYNAMIC_STORAGE_BIT); // Indicates archetype and archetype class per datapoint (0 = no archetype)
      glAssert();
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
      if(D == 3) { _trackballInputTask = std::dynamic_pointer_cast<vis::TrackballInputTask>(vis::InputQueue::instance().find("TrackballInputTask")); }
    }

    // Setup render tasks
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      _axesRenderTask = queue.emplace(vis::AxesRenderTask<D>(buffers(), _params, 1));
      _embeddingRenderTask = queue.emplace(vis::EmbeddingRenderTask<D>(_params, 0, buffers(), colorPtr));
      _selectionRenderTask = queue.emplace(vis::SelectionRenderTask(_params, 5));
      _attributeRenderTask = queue.emplace(vis::AttributeRenderTask(_params, 10, buffers(), _similaritiesBuffers, _embeddingRenderTask->getClassColorBuffer(), labelPtr));
      _linkRenderTask = queue.emplace(vis::LinkRenderTask<D>(_params, 20, buffers(), _similaritiesBuffers));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    _klDivergence = KLDivergence(_params, _similaritiesBuffers, buffers());

    _isInit = true;
    glAssert();
  }

  // Generate randomized embedding data
  // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
  template <uint D>
  void Minimization<D>::initializeEmbeddingRandomly(int seed) {
    
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

  template <uint D>
  Minimization<D>::~Minimization() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _isInit = false;
    }
  }

  template <uint D>
  Minimization<D>::Minimization(Minimization<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  Minimization<D>& Minimization<D>::operator=(Minimization<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // Restarts the minimization
  template <uint D>
  void Minimization<D>::restartMinimization() {
    if(_iteration < 100) { return; }
    if(_input.alt) { initializeEmbeddingRandomly(_iteration); } else
    if(_input.num >= 0) { initializeEmbeddingRandomly(_input.num); }
    else { initializeEmbeddingRandomly(_params->seed); }
    _iteration = 0;
    _iterationIntense = 1000;
    restartExaggeration(_params->nExaggerationIters);
    const std::vector<vec> unitvecs(_params->n, vec(1));
    glClearNamedBufferData(_buffers(BufferType::ePrevGradients), GL_R32F, GL_RED, GL_FLOAT, NULL);
    glClearNamedBufferData(_buffers(BufferType::eGain), GL_R32F, GL_RED, GL_FLOAT, unitvecs.data());
  }

  // Restarts the exaggeration by pushing the exaggeration end iteration further ahead
  template <uint D>
  void Minimization<D>::restartExaggeration(uint nExaggerationIters) {
    _removeExaggerationIter = _iteration + nExaggerationIters;
  }

  template <uint D>
  void Minimization<D>::deselect() {
    std::fill(_selectionCounts.begin(), _selectionCounts.end(), 0);
    _selectionRenderTask->setSelectionCounts(_selectionCounts);
    _attributeRenderTask->clearSelection();
    _linkRenderTask->clearLinks();
    glClearNamedBufferData(_buffers(BufferType::eSelection), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
  }

  template <uint D>
  void Minimization<D>::selectAll() {
    dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, 1, 0, _buffers(BufferType::eDisabled));
    if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
    compIterationSelect(true);
  }

  template <uint D>
  void Minimization<D>::selectInverse() {
    dh::util::BufferTools::instance().flip<int>(_buffers(BufferType::eSelection), _params->n);
    if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
    compIterationSelect(true);
  }

  template <uint D>
  void Minimization<D>::selectIndividualDatapoint(uint i) {
    uint sn = _input.s + 1; // Selection number
    glNamedBufferSubData(_buffers(BufferType::eSelection), i * sizeof(int), sizeof(int), &sn);
    compIterationSelect(true);
  }

  template <uint D>
  void Minimization<D>::stateImport() {
    std::vector<GLuint> archetypeHandles;
    std::vector<uint> archetypeClasses;
    dh::util::readState(_params->n, _params->nHighDims, D, _buffers, _similaritiesBuffers.attributeWeights, _attributeRenderTask->getWeightedAttributeIndices(), archetypeHandles, archetypeClasses);
    syncBufferHandles();
    _attributeRenderTask->mirrorWeightsToOverlay();
    _attributeRenderTask->setArchetypeHandles(archetypeHandles);
    _attributeRenderTask->setArchetypeClasses(archetypeClasses);
    _attributeRenderTask->mirrorWeightsToOverlay();
    if(_params->imageDataset) { _attributeRenderTask->copyTextureDataToTextures(); }
    compIterationSelect(true);
  }

  template <uint D>
  void Minimization<D>::stateExport() {
    dh::util::writeState(_params->n, _params->nHighDims, D, _buffers, _similaritiesBuffers.attributeWeights, _attributeRenderTask->getWeightedAttributeIndices(), _attributeRenderTask->getArchetypeHandles(), _attributeRenderTask->getArchetypeClasses());
  }

  template <uint D>
  void Minimization<D>::syncBufferHandles() {
    _similaritiesBuffers = _similarities->getBuffers();
    _embeddingRenderTask->setMinimizationBuffers(buffers()); 
    _attributeRenderTask->setMinimizationBuffers(buffers());
    _attributeRenderTask->setSimilaritiesBuffers(_similaritiesBuffers);
    _linkRenderTask->setMinimizationBuffers(buffers());
    _linkRenderTask->setSimilaritiesBuffers(_similaritiesBuffers);
  }

  template <uint D>
  void Minimization<D>::separationModeStart() {
    std::vector<uint> archetypeIndices = _attributeRenderTask->getArchetypeIndices();
    std::vector<uint> archetypeClasses = _attributeRenderTask->getArchetypeClasses();
    
    dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eFixed), _params->n, 1, 0, _buffers(BufferType::eSelection));
    dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eDisabled), _params->n, 1, 1, _buffers(BufferType::eSelection));
    deselect();
    for(uint i = 0; i < archetypeIndices.size(); ++i) {
      uint archetypeIndex = archetypeIndices[i];
       int archetypeClass = archetypeClasses[i] + 1;
      selectIndividualDatapoint(archetypeIndex);
      glNamedBufferSubData(_buffers(BufferType::eArchetypes), archetypeIndex * sizeof(int), sizeof(int), &archetypeClass);
    }
    dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eDisabled), _params->n, 0, 1, _buffers(BufferType::eSelection));
    _separationMode = true;
  }

  template <uint D>
  void Minimization<D>::separationModeStop() {
    glClearNamedBufferData(_buffers(BufferType::eFixed), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
    glClearNamedBufferData(_buffers(BufferType::eDisabled), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
    glClearNamedBufferData(_buffers(BufferType::eArchetypes), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
    float forceWeight = std::max(_embeddingRenderTask->getForceWeight() / _selectionCounts[0], 1.f);
    dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eFixed), _params->n, 1, 1, _buffers(BufferType::eSelection));
    dh::util::BufferTools::instance().set<float>(_buffers(BufferType::eWeights), _params->n, forceWeight * 2, 1, _buffers(BufferType::eSelection));
    _separationMode = false;
  }

  template <uint D>
  void Minimization<D>::comp() {
    while (_iteration < _params->iterations) {
      compIteration();
    }
  }

  // Core function handling everything that needs to happen each frame
  template <uint D>
  void Minimization<D>::compIteration() {
    _input = _selectionInputTask->getInput();

    _selectionRenderTask->setInput(_input);
    _attributeRenderTask->setInput(_input);
    _linkRenderTask->setInput(_input);

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
    if(D == 3) {
      _model_view_3D = _trackballInputTask->matrix() * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f)); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)
      _proj_3D = glm::perspectiveFov(0.5f, resolution.x, resolution.y, 0.0001f, 1000.f); // TODO: get this from Rendered (and remove trackballInputTask from Minimizatiion)  
    }

    if(_input.d) { deselect(); } // Deselect
    
    if(!_input.mouseRight) { glClearNamedBufferData(_buffers(BufferType::eTranslating), GL_R32I, GL_RED_INTEGER, GL_INT, NULL); } // Clear translations if not translating
    
    // Free/unfix fixed datapoints
    if(_input.f) {
      const std::vector<float> ones(_params->n, 1.0f);
      glClearNamedBufferData(_buffers(BufferType::eWeights), GL_R32F, GL_RED, GL_FLOAT, ones.data());
      glClearNamedBufferData(_buffers(BufferType::eFixed), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
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
      uint nEnabled = dh::util::BufferTools::instance().reduce<int>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      float fracDisabled = (float) _selectionCounts[0] / (float) nEnabled;
      dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eDisabled), _params->n, 1, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, 0, 1, _buffers(BufferType::eDisabled));
      compIterationSelect(true);
      _similarities->weighSimilarities(1.f / (1.f - fracDisabled));
      _embeddingRenderTask->setPointRadius(std::min(100.f / (nEnabled - _selectionCounts[0]), 0.005f));
    }
    if(_input.ins) { // Enable
      uint nEnabled = dh::util::BufferTools::instance().reduce<int>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      float fracEnabled = (float) nEnabled / (float) _params->n;
      _similarities->weighSimilarities(fracEnabled);
      glClearNamedBufferData(_buffers(BufferType::eDisabled), GL_R32I, GL_RED_INTEGER, GL_INT, NULL);
      _embeddingRenderTask->setPointRadius(std::min(100.f / _params->n, 0.005f));
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
        _similarities->renormalizeSimilarities();
      }
      if(button == 11) { // Add similarities
        _similarities->addSimilarities(_buffers(BufferType::eSelection), _selectionCounts, _selectionRenderTask->getSimilarityWeight());
        syncBufferHandles();
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
        _similarities->weighSimilaritiesPerAttributeResemble(weightedAttributeIndices, _buffers(BufferType::eSelection), _selectionCounts[0], _buffers(BufferType::eLabels), _attributeRenderTask->getArchetypeHandles(), _attributeRenderTask->getArchetypeClasses());
      }
      if(button == 3) { // Reset similarities
        _similarities->reset();
      }
      if(button == 1000) { // Seperation mode start
        separationModeStart();
      }
      if(button == 1001) { // Seperation mode stop
        separationModeStop();
      }
    }
    _buttonAttributePrev = button;

    // Isolate selection
    if(_embeddingRenderTask->getButtonPressed() == 1) {
      uint n = _params->n;
      _similarities->recomp(_buffers(BufferType::eSelection), _embeddingRenderTask->getPerplexity(), _embeddingRenderTask->getK());
      dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eEmbeddingRelative), n, D, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<float>(_buffers(BufferType::eWeights), n, 1, _buffers(BufferType::eSelection));
      // dh::util::BufferTools::instance().remove<int>(_buffers(BufferType::eLabels), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<int>(_buffers(BufferType::eLabels), n, 1, _buffers(BufferType::eSelection), 0, true);
      dh::util::BufferTools::instance().remove<int>(_buffers(BufferType::eLabeled), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<int>(_buffers(BufferType::eFixed), n, 1, _buffers(BufferType::eSelection));
      dh::util::BufferTools::instance().remove<int>(_buffers(BufferType::eDisabled), n, 1, _buffers(BufferType::eSelection));
      deselect();
      uint nEnabled = dh::util::BufferTools::instance().reduce<int>(_buffers(BufferType::eDisabled), 3, _params->n, 0, 0);
      _embeddingRenderTask->setPointRadius(std::min(100.f / (nEnabled - _selectionCounts[0]), 0.005f));
      syncBufferHandles(); // Update buffer handles, because BufferTools::remove() deletes and recreates buffers
      restartMinimization();
      glAssert();
    }

    int classToSelect = _attributeRenderTask->getClassButtonPressed();
    if(classToSelect >= 0) {
      dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, _input.s ? 2 : 1, (uint) classToSelect, _buffers(BufferType::eLabels));
      if(_selectOnlyLabeled) { dh::util::BufferTools::instance().set<int>(_buffers(BufferType::eSelection), _params->n, 0, 0, _buffers(BufferType::eLabeled)); }
      compIterationSelect(true);
    }

    // Select individual datapoints if SelectionRenderTask's int field input changed
    uint selectedDatapoint = (uint) _selectionRenderTask->getSelectedDatapoint();
    if(selectedDatapoint != _selectedDatapointPrevSRT && selectedDatapoint < _params->n) { selectIndividualDatapoint(selectedDatapoint); }
    _selectedDatapointPrevSRT = selectedDatapoint;

    // Select individual datapoints if AttributeRenderTask's archetype suggestions are clicked
    selectedDatapoint = _attributeRenderTask->getSelectedDatapoint();
    if(selectedDatapoint != _selectedDatapointPrevART && selectedDatapoint < _params->n) { selectIndividualDatapoint(selectedDatapoint); }
    _selectedDatapointPrevART = selectedDatapoint;

    if(_embeddingRenderTask->getButtonPressed() == 10) {
      stateImport();
    } else
    if(_embeddingRenderTask->getButtonPressed() == 11) {
      stateExport();
    }

    // if(_input.ctrl && !__assessed) {
    //   _attributeRenderTask->assess(_similarities->getSymmetricSize());
    //   __assessed = true;
    // }

    glAssert();
  }

  template <uint D>
  void Minimization<D>::compIterationMinimize() {

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
      program.template uniform<float>("weightFalloff", _embeddingRenderTask->getForceWeightFalloff());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDisabled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eArchetypes));
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

    ++_iteration;

    // Log progress; spawn progressbar on the current (new on first iter) line
    // reporting current iteration and size of field texture
    if (!_loggedNewline) {
      _loggedNewline = true;
    }
    if (_params->verbose && (_iteration % 100) == 0) {
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

  template <uint D>
  void Minimization<D>::compIterationSelect(bool skipEval) {
    uint si = _input.s; // Selection index
    uint sn = si + 1; // Selection number

    // 1.
    // Compute selection (or select all)
    if(!skipEval) {
      auto& program = _programs(ProgramType::eSelectionComp);
      program.bind();

      // Set uniform
      program.template uniform<uint>("nPoints", _params->n);
      program.template uniform<int>("selectionNumber", sn);
      program.template uniform<float, 2>("mousePosClip", _input.mousePosClip);
      program.template uniform<float>("selectionRadiusRel", _selectionRadiusRel);
      program.template uniform<float>("selectOnlyLabeled", _selectOnlyLabeled);
      program.template uniform<float, 4, 4>("model_view", D == 2 ? _model_view_2D : _model_view_3D);
      program.template uniform<float, 4, 4>("proj", D == 2 ? _proj_2D : _proj_3D);
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
      _selectionCounts[s] = dh::util::BufferTools::instance().reduce<int>(_buffers(BufferType::eSelection), 3, _params->n, 0, s + 1);

      _selectionRenderTask->setSelectionCounts(_selectionCounts);
    }

    _attributeRenderTask->updateVisualizations(_selectionCounts);
    _linkRenderTask->setSecondarySelectionCount(_selectionCounts[1]);
    _linkRenderTask->updateLinks();
  }

  template <uint D>
  void Minimization<D>::compIterationTranslate() {

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
    program.template uniform<float>("forceWeight", std::max(_embeddingRenderTask->getForceWeight() / _selectionCounts[0], 1.f));

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
  template class Minimization<2>;
  template class Minimization<3>;
} // dh::sne