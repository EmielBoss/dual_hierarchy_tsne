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
#include "dh/vis/components/embedding_render_task.hpp"
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

  template<typename Type>
  void checkBuffer(GLuint handle, uint n, bool d, uint s) {
    std::vector<Type> buffer(n);
    glGetNamedBufferSubData(handle, 0, n * sizeof(Type), buffer.data());
    float avg = 0;
    for(uint i = 0; i < n; ++i) {
      avg += std::abs(buffer[i]);
    }
    // avg /= (float) n;
    std::cout << d;
  }

  template <uint D>
  Minimization<D>::Minimization(Similarities* similarities, Params params)
  : _isInit(false), _similarities(similarities), _similaritiesBuffers(similarities->buffers()), _params(params), _iteration(0), _selectionIdx(0) {
    Logger::newt() << prefix << "Initializing...";

    _selectionCounts[0] = 0;
    _selectionCounts[1] = 0;

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
        _programs(ProgramType::eSelectionCountComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/selection_count.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
        _programs(ProgramType::eSelectionCountComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/selection_count.comp"));
      }
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      std::vector<uint> selection(_params.n);
      std::ifstream fin("../selection.txt", std::ios::in);
      char byte = 0;
      for (uint i = 0; fin.get(byte); ++i) {
        selection[i] = byte - '0';
      }
      fin.close();
      glAssert();

      const std::vector<vec> zeroes(_params.n, vec(0));
      const std::vector<vec> ones(_params.n, vec(1));
      const std::vector<uint> falses(_params.n, 0);

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params.n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _params.n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), _params.n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), _params.n * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborsEmb), _params.n * _params.k * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eDistancesEmb), _params.n * _params.k * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighborhoodPreservation), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelection), _params.n * sizeof(uint), falses.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eSelectionCounts), 2 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSelectionCountsReduce), 128 * sizeof(uint), nullptr, 0);
      glAssert();
    }

    // Generate randomized embedding data
    // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
    {
      // Seed the (bad) rng
      std::srand(_params.seed);
      
      // Generate n random D-dimensional vectors
      std::vector<vec> embedding(_params.n, vec(0.f));
      for (uint i = 0; i < _params.n; ++i) {
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

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Setup field subcomponent
    _field = Field<D>(buffers(), _params);

#ifdef DH_ENABLE_VIS_EMBEDDING
    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      _embeddingRenderTask = queue.emplace(vis::EmbeddingRenderTask<D>(buffers(), _params, 0));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    // Get selectionInputTask subcomponent for mouse input for selecting
    _selectionInputTask = std::dynamic_pointer_cast<vis::SelectionInputTask>(vis::InputQueue::instance().find("SelectionInputTask"));

    _selectionRenderTask = std::dynamic_pointer_cast<vis::SelectionRenderTask>(vis::RenderQueue::instance().find("SelectionRenderTask"));

    _isInit = true;
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

  template <uint D>
  void Minimization<D>::comp() {
    while (_iteration < _params.iterations) {
      compIteration();
    }
  }

    template <uint D>
  void Minimization<D>::compIteration() {
    _dPressed = _selectionInputTask->getDPressed(); // For debugging purposes
    _mousePressed = _selectionInputTask->getMousePressed();
    _spacePressed = _selectionInputTask->getSpacePressed();
    if(!_spacePressed                     ) { compIterationMinimization(); }
    if( _mousePressed || _mousePressedPrev) { compIterationSelection(); }
    _mousePressedPrev = _mousePressed;
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
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eBounds));

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

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similaritiesBuffers.layout);  // n structs of two uints; the first is the offset into _similaritiesBuffers.neighbors where its kNN set starts, the second is the size of its kNN set
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similaritiesBuffers.neighbors); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similaritiesBuffers.similarities); // Corresponding similarities
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1); // One warp/subgroup per i
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

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGain));

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

    // This is super stupid, but its a way to synchronize colorMapping between the two input methods (GUI and numkeys)
    int num = _selectionInputTask->getNumPressed();
    if(num != _colorMapping && num >= 0) { _embeddingRenderTask->setColorMapping(num); }
    num = _embeddingRenderTask->getColorMapping();
    if (num != _colorMapping && num >= 0) { _selectionInputTask->setNumPressed(num); }

    if(num == 2 && _colorMapping != 2) {
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
        program.template uniform<uint>("k", _params.k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _similaritiesBuffers.layout);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similaritiesBuffers.neighbors);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighborsEmb));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eNeighborhoodPreservation));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      glAssert();
    }
    _colorMapping = num;


    std::vector<uint> layout(2);
    glGetNamedBufferSubData(_similaritiesBuffers.layout, 2 * (_params.n - 1) * sizeof(uint), 2, layout.data());
    std::vector<float> neighboursEmb(layout[0] + layout[1]);
    glGetNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, (layout[0] + layout[1]) * sizeof(uint), neighboursEmb.data());

    // Log progress; spawn progressbar on the current (new on first iter) line
    // reporting current iteration and size of field texture
    if (_iteration == 0) {
      Logger::newl();
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
    uint selectionNumber = _selectionIdx + 1;
    
    // 1.
    // Compute selection
    {
      auto& program = _programs(ProgramType::eSelectionComp);
      program.bind();

      // Get everything related with the cursor and selection brush
      const vec2 boundsCenter = {_bounds.center().x, _bounds.center().y};
      const vec boundsRange = _bounds.range();
      const vec2 boundsRangeHalf = {boundsRange.x, boundsRange.y / 1.8f}; // No clue why, but this works. Plz don't hurt me.
      vec2 boundsMin = {_bounds.min.x, _bounds.min.y}; // Only need the first two dimensions in case D == 3
      vec2 boundsMax = {_bounds.max.x, _bounds.max.y};
      vec2 mousePos = _selectionInputTask->getMousePos();
      vec2 cursorPos = boundsCenter + boundsRangeHalf * mousePos;
      float selectionRadiusPixel = _selectionRenderTask->getSelectionRadius();
      float selectionRadius = boundsRange.y * selectionRadiusPixel / _params.resHeight;

      // Set uniform
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<float, 2>("cursorPos", cursorPos);
      program.template uniform<float>("selectionRadius", selectionRadius);
      program.template uniform<uint>("selectionNumber", selectionNumber);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelection));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }
    
    // 2.
    // Things to do when at the end of any selection
    if(!_mousePressed && _mousePressedPrev) {

      // Count number of selected datapoints in the just-finished selection
      {
        
        auto& program = _programs(ProgramType::eSelectionCountComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelection));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectionCountsReduce));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectionCounts));
        glAssert();

        for(uint s = 1; s < selectionNumber + 1; s++) {
          glClearNamedBufferData(_buffers(BufferType::eSelectionCountsReduce), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
          program.template uniform<uint>("selectionNumber", s);
          program.template uniform<uint>("iter", 0);
          glDispatchCompute(128, 1, 1);
          glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
          program.template uniform<uint>("iter", 1);
          glDispatchCompute(1, 1, 1);
          glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        glAssert();
      }

      glGetNamedBufferSubData(_buffers(BufferType::eSelectionCounts), 0, 2 * sizeof(uint), _selectionCounts);

      if(_selectionCounts[_selectionIdx] < 5) { return; }
      _selectionIdx = ++_selectionIdx % 2; // Increment selection index
      glAssert();

    }

    // 3.
    // Update neighbors at the end of second selection

    if(!_mousePressed && _mousePressedPrev && selectionNumber == 2) {

      {
        glClearNamedBufferData(_buffers(BufferType::eSelectionCountsReduce), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        
        auto& program = _programs(ProgramType::eSelectionCountComp);
        program.bind();

        // Set uniforms
        program.template uniform<uint>("nPoints", _params.n);
        program.template uniform<uint>("selectionNumber", selectionNumber);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSelection));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSelectionCountsReduce));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSelectionCounts));
        glAssert();

        program.template uniform<uint>("iter", 0);
        glDispatchCompute(128, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        program.template uniform<uint>("iter", 1);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        glAssert();
      }

      glGetNamedBufferSubData(_buffers(BufferType::eSelectionCounts), 0, 2 * sizeof(uint), &_selectionCounts);

      // _similarities->update(_buffers(BufferType::eSelection), _buffers(BufferType::eSelectionCounts), _selectionCounts);
      // _similaritiesBuffers = _similarities->buffers(); // Update buffer handles
      glClearNamedBufferData(_buffers(BufferType::eSelection), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eSelectionCounts), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr); // Clear counts
      glAssert();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class Minimization<2>;
  template class Minimization<3>;
} // dh::sne