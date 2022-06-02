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
#include <resource_embed/resource_embed.hpp>
#include "dh/constants.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/util/cu/knn.cuh"
#include <numeric> //
#include <fstream> //
#include <filesystem> //

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
  void Minimization<D>::writeBuffer(GLuint handle, uint n, uint d, std::string filename) {
    std::vector<float> buffer(n * d);
    glGetNamedBufferSubData(handle, 0, n * d * sizeof(float), buffer.data());
    std::ofstream file("../buffer_dumps/" + filename + ".txt");
    for(uint i = 0; i < n; i++) {
      for(uint j = 0; j < d; ++j) {
        file << buffer[i * d + j] << "|";
      }
      file << "\n";
    }
  }

  template <uint D>
  Minimization<D>::Minimization(Similarities* similarities, const float* dataPtr, const int* labelPtr, Params params)
  : _isInit(false), _loggedNewline(false), _similarities(similarities), _similaritiesBuffers(similarities->buffers()), _selectionCount(0), _params(params), _iteration(0) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {      
      if constexpr (D == 2) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/gradients.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/centerEmbedding.comp"));
        _programs(ProgramType::eNeighborhoodPreservationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/neighborhood_preservation.comp"));
        _programs(ProgramType::eSelectionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selection.comp"));
        _programs(ProgramType::eCountSelectedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selected_count.comp"));
        _programs(ProgramType::eAverageSelectedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/selected_average.comp"));
        _programs(ProgramType::eTranslationComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/translation.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
      }
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      std::vector<uint> labeledIndices = {0, 1, 2, 3, 4, 5, 7, 13, 15, 17}; // Purely for development and demonstration on MNIST
      const std::vector<vec> zerovecs(_params.n, vec(0));
      const std::vector<vec> unitvecs(_params.n, vec(1));
      const std::vector<uint> falses(_params.n, 0); // TODO: use bools instead of uints
      const std::vector<float> ones(_params.n, 1.0f);
      std::vector<uint> labeled(_params.n, 0);
      // for(uint i = 0; i < _params.n; ++i) { if (*(labelPtr + i) >= 0)  { labeled[i] = 1; } }
      for(uint i = 0; i < labeledIndices.size(); ++i) { labeled[labeledIndices[i]] = 1; }
      std::vector<float> data(_params.n * _params.nHighDims);
      for(uint p = 0; p < _params.n * _params.nHighDims; ++p) { data[p] = *(dataPtr + p) / 255.0f; }

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params.n * _params.nHighDims * sizeof(float), data.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params.n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), unitvecs.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _params.n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), _params.n * sizeof(vec), zerovecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), _params.n * sizeof(vec), unitvecs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborsEmb), _params.n * _params.k * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eDistancesEmb), _params.n * _params.k * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborhoodPreservation), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelected), _params.n * sizeof(uint), falses.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectedCount), sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectedCountReduce), 128 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectedAverage), _params.nHighDims * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectedAverageReduce), 128 * _params.nHighDims * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eFixed), _params.n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eTranslating), _params.n * sizeof(uint), falses.data(), 0); // Indicates whether datapoints are being translated
      glNamedBufferStorage(_buffers(BufferType::eWeights), _params.n * sizeof(float), ones.data(), 0); // The attractive force multiplier per datapoint
      glNamedBufferStorage(_buffers(BufferType::eLabeled), _params.n * sizeof(uint), labeled.data(), 0); // Indicates whether datapoints are fixed
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelative), _params.n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingRelativeBeforeTranslation), _params.n * sizeof(vec), nullptr, 0);
      glAssert();

      glCreateTextures(GL_TEXTURE_2D, 1, &_averageSelectionTexture);
      glTextureParameteri(_averageSelectionTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_averageSelectionTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureStorage2D(_averageSelectionTexture, 1, GL_R8, _params.imgWidth, _params.imgHeight);
    }

    initializeEmbeddingRandomly(_params.seed);

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Setup field subcomponent
    _field = Field<D>(buffers(), _params);

#ifdef DH_ENABLE_VIS_EMBEDDING
    // Setup render tasks
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      _embeddingRenderTask = queue.emplace(vis::EmbeddingRenderTask<D>(buffers(), _params, 0));
      _selectionRenderTask = queue.emplace(vis::SelectionRenderTask(buffers(), _params, 5, dataPtr));
      _borderRenderTask = queue.emplace(vis::BorderRenderTask<D>(buffers(), _params, 1));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    // Get selectionInputTask subcomponent for mouse input for selecting
    _selectionInputTask = std::dynamic_pointer_cast<vis::SelectionInputTask>(vis::InputQueue::instance().find("SelectionInputTask"));

    _isInit = true;
  }

  // Generate randomized embedding data
  // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
  template <uint D>
  void Minimization<D>::initializeEmbeddingRandomly(int seed) {
    
    // Copy over embedding and fixed buffers to host
    std::vector<vec> embedding(_params.n);
    glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params.n * sizeof(vec), embedding.data());
    std::vector<uint> fixed(_params.n); // These are all zeroes on first initialization anyways
    glGetNamedBufferSubData(_buffers(BufferType::eFixed), 0, _params.n * sizeof(uint), fixed.data());

    // Seed the (bad) rng
    std::srand(seed);
    
    // Generate n random D-dimensional vectors
    for (uint i = 0; i < _params.n; ++i) {
      if(fixed[i]) { continue; }
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
      embedding[i] = v * r * _params.rngRange;
    }

    // Copy to buffer
    glNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, _params.n * sizeof(vec), embedding.data());
    glAssert();
  }

  template <uint D>
  Minimization<D>::~Minimization() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteTextures(1, &_averageSelectionTexture);
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

  template <uint D>
  void Minimization<D>::comp() {
    while (_iteration < _params.iterations) {
      compIteration();
    }
  }

  template <uint D>
  void Minimization<D>::compIteration() {
    glm::vec2 mousePosClipPrev = _input.mousePosClip;
    _mouseLeftPrev = _input.mouseLeft;
    _mouseRightPrev = _input.mouseRight;
    _input = _selectionInputTask->getInput();
    bool mouseRight = _input.mouseRight;

    // Synchronizing color mapping between GUI and input
    _colorMappingPrev = _colorMapping;
    _colorMapping = _input.num;
    if(_colorMapping != _colorMappingPrev) { _embeddingRenderTask->setColorMapping(_colorMapping); }
    _colorMapping = _embeddingRenderTask->getColorMapping();
    if (_colorMapping != _colorMappingPrev) { _selectionInputTask->setNumPressed(_colorMapping); }

    // Synchronizing selection radius between GUI and input
    float selectionRadiusRelPrev = _selectionRadiusRel;
    _selectionRadiusRel = _selectionRenderTask->getSelectionRadiusRel();
    if (_selectionRadiusRel != selectionRadiusRelPrev) { _selectionInputTask->setMouseScroll(_selectionRadiusRel * 100); }
    _selectionRadiusRel = _input.mouseScroll / 100.f;
    if(_selectionRadiusRel != selectionRadiusRelPrev) { _selectionRenderTask->setSelectionRadiusRel(_selectionRadiusRel); }

    _selectionRenderTask->setMousePosition(_input.mousePosScreen);
    // Synchronize selection mode
    _selectOnlyLabeledPrev = _selectOnlyLabeled;
    _selectOnlyLabeled = _selectionRenderTask->getSelectionMode();
    _embeddingRenderTask->setSelectionMode(_selectOnlyLabeled);

    // Get everything related with the cursor and selection brush
    const glm::vec2 boundsRange = {_bounds.range().x, _bounds.range().y};
    const glm::vec2 boundsMin = {_bounds.min.x, _bounds.min.y};

    glm::mat4 model_view = glm::translate(glm::vec3(-0.5f, -0.5f, -1.0f));
    glm::mat4 proj = glm::infinitePerspective(1.0f, _selectionRenderTask->getAspectRatio(), 0.0001f);
    glm::vec4 mousePosClipInverted = glm::inverse(proj * model_view) * glm::vec4(_input.mousePosClip.x, _input.mousePosClip.y, 0.9998, 1);
    _mousePosEmbedding = glm::vec2(mousePosClipInverted.x, mousePosClipInverted.y) * boundsRange + boundsMin;
    glm::vec4 mousePosClipPrevInverted = glm::inverse(proj * model_view) * glm::vec4(mousePosClipPrev.x, mousePosClipPrev.y, 0.9998, 1);
    _mousePosEmbeddingPrev = glm::vec2(mousePosClipPrevInverted.x, mousePosClipPrevInverted.y) * boundsRange + boundsMin;

    if(_input.mouseMiddle || (_selectOnlyLabeled != _selectOnlyLabeledPrev)) {
      _selectionCount = 0;
      glClearNamedBufferData(_buffers(BufferType::eSelected), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearTexImage(_averageSelectionTexture, 0,  GL_RED, GL_FLOAT, nullptr);
      _embeddingRenderTask->setWeighForces(true);
    }
    if(!mouseRight) { glClearNamedBufferData(_buffers(BufferType::eTranslating), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); }
    if(_input.f) {
      const std::vector<float> ones(_params.n, 1.0f);
      glClearNamedBufferData(_buffers(BufferType::eWeights), GL_R32F, GL_RED, GL_FLOAT, ones.data());
      glClearNamedBufferData(_buffers(BufferType::eFixed), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    }

    if(_input.r) { compIterationMinimizationRestart(); }
    if(!_input.space) { compIterationMinimization(); }
    if(_input.mouseLeft  ) { compIterationSelection(); }
    if(_input.mouseRight || _mouseRightPrev) { compIterationTranslation(); }
  }

  template <uint D>
  void Minimization<D>::compIterationMinimizationRestart() {
    if(_iteration < 100) { return; }
    initializeEmbeddingRandomly(_iteration);
    _iteration = 0;
    const std::vector<vec> zerovecs(_params.n, vec(0));
    glClearNamedBufferData(_buffers(BufferType::ePrevGradients), GL_R32F, GL_RED, GL_FLOAT, zerovecs.data());
  }

  template <uint D>
  void Minimization<D>::compIterationMinimization() {

    // 1.
    // Compute embedding bounds
    {
      auto& timer = _timers(TimerType::eBoundsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eBoundsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("padding", 0.0f);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eTranslating));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eBounds));

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
      const float ratio = (D == 2) ? _params.fieldScaling2D : _params.fieldScaling3D;
      uvec size = dh::util::max(uvec(range * ratio), uvec(fieldMinSize));

      // Size becomes nearest larger power of two for field hierarchy
      size = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(size.x)) / glm::log(2.f))));

      // Delegate to subclass
      _field.comp(size, _iteration, _embeddingRenderTask->getWeightFixed());
    }

    // 3.
    // Compute Z, ergo a reduction over q_{ij}
    {
      auto& timer = _timers(TimerType::eZComp);
      timer.tick();

      auto& program = _programs(ProgramType::eZComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eZReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));

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
      program.template uniform<uint>("nPos", _params.n);
      program.template uniform<float>("invPos", 1.f / static_cast<float>(_params.n));
      program.template uniform<bool>("weighForces", _embeddingRenderTask->getWeighForces());
      program.template uniform<float>("weightFalloff", _embeddingRenderTask->getWeightFalloff());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.layout);  // n structs of two uints; the first is the offset into _similaritiesBuffers.neighbors where its kNN set starts, the second is the size of its kNN set
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similaritiesBuffers.neighbors); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _similaritiesBuffers.similarities); // Corresponding similarities
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1); // One warp/subgroup per datapoint
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Compute exaggeration factor
    float exaggeration = 1.0f;
    if (_iteration <= _params.removeExaggerationIter) {
      exaggeration = _params.exaggerationFactor;
    } else if (_iteration <= _params.removeExaggerationIter + _params.exponentialDecayIter) {
      float decay = 1.0f - static_cast<float>(_iteration - _params.removeExaggerationIter)
                         / static_cast<float>(_params.exponentialDecayIter);
      exaggeration = 1.0f + (_params.exaggerationFactor - 1.0f) * decay;
    }

    // 5.
    // Compute gradients
    {
      auto& timer = _timers(TimerType::eGradientsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eGradientsComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttractive));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Precompute instead of doing it in shader N times
    const float iterMult = (static_cast<double>(_iteration) < _params.momentumSwitchIter) 
                         ? _params.momentum 
                         : _params.finalMomentum;

    // 6.
    // Update embedding
    {
      glm::vec2 boundsScaling = _bounds.range() / _boundsPrev.range();
      glm::vec2 boundsTranslation = _bounds.center() - _boundsPrev.center();

      auto& timer = _timers(TimerType::eUpdateEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eUpdateEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("eta", _params.eta);
      program.template uniform<float>("minGain", _params.minimumGain);
      program.template uniform<float>("mult", 1.0);
      program.template uniform<float>("iterMult", iterMult);
      program.template uniform<bool>("translationFinished", !_input.mouseRight && _mouseRightPrev);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eGain));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
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
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float>("scaling", scaling);
      program.template uniform<float, D>("center", boundsCenter);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 8.
    // Compute neighborhood preservation per datapoint

    if(_colorMapping == 2 && _colorMappingPrev != 2) {
      // Compute approximate KNN of each point in embedding, delegated to FAISS
      std::vector<float> embedding(2 * _params.n);
      glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, 2 * _params.n * sizeof(float), embedding.data());
      {
        util::KNN knn(
          embedding.data(),
          _buffers(BufferType::eDistancesEmb),
          _buffers(BufferType::eNeighborsEmb),
          _params.n, _params.k, D);
        knn.comp();
      }

      {
        auto& program = _programs(ProgramType::eNeighborhoodPreservationComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);
        program.template uniform<uint>("kParam", _params.k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _similaritiesBuffers.layout);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similaritiesBuffers.neighbors);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighborsEmb));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighborhoodPreservation));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        glAssert();
      }
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
      
      const std::string postfix = (_iteration < _params.iterations)
                                ? "iter: " + std::to_string(_iteration) + ", field: " + fieldStr.str()
                                : "Done!";
      util::ProgressBar progressBar(prefix + "Computing...", postfix);
      progressBar.setProgress(static_cast<float>(_iteration) / static_cast<float>(_params.iterations));
    }
  }

  template <uint D>
  void Minimization<D>::compIterationSelection() {

    // 1.
    // Compute selection
    {
      auto& program = _programs(ProgramType::eSelectionComp);
      program.bind();

      float selectionRadiusEmbedding = _bounds.range().y * _selectionRadiusRel;

      // Set uniform
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float, 2>("cursorPos", _mousePosEmbedding);
      program.template uniform<float>("selectionRadius", selectionRadiusEmbedding);
      program.template uniform<float>("selectOnlyLabeled", _selectOnlyLabeled);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLabeled));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelected));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    // 2.
    // Count number of selected datapoints
    {
      auto& program = _programs(ProgramType::eCountSelectedComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelected));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectedCountReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectedCount));
      glAssert();

      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      uint selectionCountPrev = _selectionCount;
      glGetNamedBufferSubData(_buffers(BufferType::eSelectedCount), 0, sizeof(uint), &_selectionCount);
      glAssert();

      // Turn of force weighing if too many datapoints are selected at once, which is likely not what the user wants
      if(_selectionCount - selectionCountPrev > _params.n / 1000) { _embeddingRenderTask->setWeighForces(false); }
    }

    // 3.
    // Average selected images
    if(_params.datapointsAreImages) {
      auto& program = _programs(ProgramType::eAverageSelectedComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("nSelected", _selectionCount);
      program.template uniform<uint>("imgSize", _params.nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelected));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectedAverageReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSelectedAverage));
      glAssert();

      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, _params.nHighDims, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1,_params.nHighDims, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _buffers(BufferType::eSelectedAverage));
      glTextureSubImage2D(_averageSelectionTexture, 0, 0, 0, _params.imgWidth, _params.imgHeight, GL_RED, GL_FLOAT, 0);
      glAssert();
    }
  }

  template <uint D>
  void Minimization<D>::compIterationTranslation() {

    // Compute translation
    {
      auto& program = _programs(ProgramType::eTranslationComp);
      program.bind();

      // Set uniform
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float, 2>("cursorPos", _mousePosEmbedding);
      program.template uniform<float, 2>("cursorPosPrev", _mousePosEmbeddingPrev);
      program.template uniform<bool>("translationStarted", !_mouseRightPrev);
      program.template uniform<bool>("translationFinished", !_input.mouseRight && _mouseRightPrev);
      program.template uniform<float>("weightFixed", _embeddingRenderTask->getWeightFixed());

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelected));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingRelative));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eEmbeddingRelativeBeforeTranslation));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eFixed));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eTranslating));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

  }

  // Template instantiations for 2/3 dimensions
  template class Minimization<2>;
  template class Minimization<3>;
} // dh::sne