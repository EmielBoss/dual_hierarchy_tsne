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

#include <resource_embed/resource_embed.hpp>
#include "dh/sne/components/similarities.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/cu/inclusive_scan.cuh"
#include "dh/util/cu/knn.cuh"
#include <typeinfo> //
#include <numeric> //
#include <fstream> //
#include <filesystem> //
#include <imgui.h> //
#include <imgui_impl_glfw.h> //
#include <imgui_impl_opengl3.h> //
#include <implot.h> //

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");

  float Similarities::average(std::vector<float> vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.f) / vec.size();
  }

  // Auxiliary function to create a window and display a scatterplot
  void Similarities::displayGraph(std::vector<float> inter, std::vector<float> intra, bool relative) {
    std::vector<float> concat = intra;
    concat.insert(concat.end(), inter.begin(), inter.end());

    dh::util::GLWindowInfo info;
    {
      using namespace dh::util;
      info.title = "Graphs";
      info.width = 1500;
      info.height = 1000;
      info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                  | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                  | GLWindowInfo::bFloating;
    }
    dh::util::GLWindow window(info);

    ImGuiContext* ctxMain = ImGui::GetCurrentContext();
    IMGUI_CHECKVERSION();
    ImGuiContext* ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *) window.handle(), true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();

    window.setVsync(true);
    window.setVisible(true);

    std::cout << "Inter count: " << inter.size() << "\n";
    std::cout << "Intra count: " << intra.size() << "\n";

    ImPlotRange range = ImPlotRange(0, 1);
    if(inter.size() > 0 && intra.size() > 0) {
      auto [minInter, maxInter] = std::minmax_element(inter.begin(), inter.end());
      auto [minIntra, maxIntra] = std::minmax_element(intra.begin(), intra.end());
      ImPlotRange range = ImPlotRange(std::min(*minInter, *minIntra), std::max(*maxInter, *maxIntra));
    }

    int nBins = 100;

    while(window.canDisplay()) {
      window.processEvents();
      glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_NewFrame(); // Start new frame for IMGUI
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Appearing);
      ImGui::SetNextWindowSize(ImVec2(1500, 1500));
      ImGui::Begin("Graphs", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

      ImGui::SliderInt("No. of bins", &nBins, 1, 1000);

      if (ImPlot::BeginPlot("Inter/intra")) {
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(0), 0.5f);
        if(inter.size() > 0) { ImPlot::PlotHistogram("Inter", inter.data(), inter.size(), nBins, relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::SetNextFillStyle(ImPlot::GetColormapColor(1), 0.5f);
        if(intra.size() > 0) { ImPlot::PlotHistogram("Intra", intra.data(), intra.size(), nBins, relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::EndPlot();
      }

      if (ImPlot::BeginPlot("Combined")) {
        if(concat.size() > 0) { ImPlot::PlotHistogram("All", concat.data(), concat.size(), nBins, relative ? ImPlotHistogramFlags_Density : 0); }
        ImPlot::EndPlot();
      }

      ImGui::End();

      ImGui::Render(); // Finalize and render ImGui components
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      window.display();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    ImGui::SetCurrentContext(ctxMain);
    glAssert();
  }

  // Auxiliary function purely for debugging; will be removed
  template<typename T>
  void Similarities::writeBuffer(GLuint handle, uint n, uint d, std::string filename) {
    std::vector<T> buffer(n * d);
    glGetNamedBufferSubData(handle, 0, n * d * sizeof(T), buffer.data());
    std::ofstream file("./buffer_dumps/" + filename + ".txt");
    for(uint i = 0; i < n; i++) {
      for(uint j = 0; j < d; ++j) {
        T val = buffer[i * d + j];
        file << val << "|";
      }
      file << "\n";
    }
  }
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const float* dataPtr, Params params)
  : _isInit(false), _dataPtr(dataPtr), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programse
    {
      _programs(ProgramType::eAccumulatePerDatapointFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/accumulate_per_datapoint_float.comp"));
      _programs(ProgramType::eAccumulatePerDatapointUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/accumulate_per_datapoint_uint.comp"));
      _programs(ProgramType::eReduceFloatComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/reduce_float.comp"));
      _programs(ProgramType::eReduceUintComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/reduce_uint.comp"));
      _programs(ProgramType::eSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/similarities.comp"));
      _programs(ProgramType::eExpandComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/expand.comp"));
      _programs(ProgramType::eLayoutComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/layout.comp"));
      _programs(ProgramType::eNeighborsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors.comp"));
      _programs(ProgramType::eNeighborsSortComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors_sort.comp"));
      _programs(ProgramType::eWeighSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities.comp"));
      _programs(ProgramType::eWeighSimilaritiesPerAttributeComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/weigh_similarities_per_attr.comp"));

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Create and initialize buffers
    glCreateBuffers(_buffers.size(), _buffers.data());
    {
      const std::vector<float> ones(_params.nHighDims, 1.0f);
      glNamedBufferStorage(_buffers(BufferType::eDataset), _params.n * _params.nHighDims * sizeof(float), _dataPtr, 0); // Original dataset. TODO: make FAISS work from this instead of _dataPtr itself to avoid copying to GPU twice
      glNamedBufferStorage(_buffers(BufferType::eScan), _params.n * sizeof(uint), nullptr, 0); // Prefix sum/inclusive scan over expanded neighbor set sizes (eSizes). (This should be a temp buffer, but that yields an error)
      glNamedBufferStorage(_buffers(BufferType::eLayout), _params.n * 2 * sizeof(uint), nullptr, 0); // n structs of two uints; the first is its expanded neighbor set offset (eScan[i - 1]), the second is its expanded neighbor set size (eScan[i] - eScan[i - 1])
      glNamedBufferStorage(_buffers(BufferType::eAttributeWeights), _params.nHighDims * sizeof(float), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glAssert();
    }

    _isInit = true;
    Logger::rest() << prefix << "Initialized";
  }

  Similarities::~Similarities() {
    if (isInit()) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  Similarities::Similarities(Similarities&& other) noexcept {
    swap(*this, other);
  }

  Similarities& Similarities::operator=(Similarities&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  // Sum reduction on a buffer. largeBuffer means larger than _params.n, in which case its contents are first accumulated per (selected) datapoint
  template<typename T>
  T Similarities::reduce(GLuint bufferToReduce, bool largeBuffer, GLuint selectionBufferHandle) {
    uint nBuffers = largeBuffer ? 3 : 2;
    glCreateBuffers(nBuffers, &_buffersReduce(BufferReduceType::eReduce));
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduce), 128 * sizeof(T), nullptr, 0);
    glNamedBufferStorage(_buffersReduce(BufferReduceType::eReduced), sizeof(T), nullptr, 0);

    if(largeBuffer) {
      glNamedBufferStorage(_buffersReduce(BufferReduceType::eAccumulationPerDatapoint), _params.n * sizeof(T), nullptr, 0);

      dh::util::GLProgram& program = std::is_same<T, float>::value
                                    ? _programs(ProgramType::eAccumulatePerDatapointFloatComp)
                                    : _programs(ProgramType::eAccumulatePerDatapointUintComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<bool>("selectedOnly", selectionBufferHandle > 0);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufferToReduce);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
      glAssert();
      std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint));
    }

    dh::util::GLProgram& program = std::is_same<T, float>::value ? _programs(ProgramType::eReduceFloatComp) : _programs(ProgramType::eReduceUintComp);
    program.bind();
  
    program.template uniform<uint>("nPoints", _params.n);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferToReduce);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersReduce(BufferReduceType::eReduce));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersReduce(BufferReduceType::eReduced));

    // Dispatch shader
    program.template uniform<uint>("iter", 0);
    glDispatchCompute(128, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.template uniform<uint>("iter", 1);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    T reducedValue = 0;
    glGetNamedBufferSubData(_buffersReduce(BufferReduceType::eReduced), 0, sizeof(T), &reducedValue);
    if(largeBuffer) { std::swap(bufferToReduce, _buffersReduce(BufferReduceType::eAccumulationPerDatapoint)); }
    glDeleteBuffers(nBuffers, &_buffersReduce(BufferReduceType::eReduce));
    glAssert();
    return reducedValue;
  }

  void Similarities::comp() {
    runtimeAssert(isInit(), "Similarities::comp() called without proper initialization");

    // Create and initialize temporary buffer objects
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    {
      const std::vector<uint> zeroes(_params.n * _params.k, 0);
      glNamedBufferStorage(_buffersTemp(BufferTempType::eDistances), _params.n * _params.k * sizeof(float), nullptr, 0); // n * k floats of neighbor distances; every k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eNeighbors), _params.n * _params.k * sizeof(uint), nullptr, 0); // n * k uints of neighbor indices (ranging from 0 to n-1); every k'th element is vector index itself (so it's actually k-1 NN)
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSizes), _params.n * sizeof(uint), zeroes.data(), 0); // n uints of (expanded) neighbor set sizes; every element is k-1 plus its number of "unregistered neighbors" that have it as neighbor but that it doesn't reciprocate
      glNamedBufferStorage(_buffersTemp(BufferTempType::eSimilarities), _params.n * _params.k * sizeof(float), zeroes.data(), 0); // n * k floats of neighbor similarities; every k'th element is 0
      glNamedBufferStorage(_buffersTemp(BufferTempType::eCounts), _params.n * sizeof(uint), zeroes.data(), 0);
    }
    
    // Progress bar for logging steps of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");
    progressBar.setPostfix("Performing KNN search");
    progressBar.setProgress(0.0f);

    // 1.
    // Compute approximate KNN of each point, delegated to FAISS
    // Produces a fixed number of neighbors
    {
      util::KNN knn(
        _dataPtr,
        _buffersTemp(BufferTempType::eDistances),
        _buffersTemp(BufferTempType::eNeighbors),
        _params.n, _params.k, _params.nHighDims);
      knn.comp();
    }

    // Update progress bar
    progressBar.setPostfix("Performing similarity computation");
    progressBar.setProgress(1.0f / 6.0f);

    // 2.
    // Compute similarities over generated KNN. This is pretty much a direct copy of the formulation used in BH-SNE, and seems to also be used in CUDA-tSNE.
    {
      auto& timer = _timers(TimerType::eSimilaritiesComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eSimilaritiesComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);
      program.template uniform<float>("perplexity", _params.perplexity);
      program.template uniform<uint>("nIters", 200); // Number of binary search iterations for finding sigma corresponding to perplexity
      program.template uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Symmetrizing KNN data");
    progressBar.setProgress(2.0f / 6.0f);

    // 3.
    // Expand KNN data so it becomes symmetric. That is, every neigbor referred by a point itself refers to that point as a neighbor.
    // Actually just fills eSizes, which, for each element, is _params.k-1+the number of "unregistered neighbors"; datapoints that have it as a neighbor but which it doesn't reciprocate
    {
      auto& timer = _timers(TimerType::eExpandComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eExpandComp);
      program.bind();
      
      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Allocating buffers");
    progressBar.setProgress(3.0f / 6.0f);

    // 4.
    // Determine sizes of expanded neighborhoods in memory through prefix sum (https://en.wikipedia.org/wiki/Prefix_sum). Leverages CUDA CUB library underneath

    {
      util::InclusiveScan scan(_buffersTemp(BufferTempType::eSizes), _buffers(BufferType::eScan), _params.n);
      scan.comp();
      glGetNamedBufferSubData(_buffers(BufferType::eScan), (_params.n - 1) * sizeof(uint), sizeof(uint), &_symmetricSize); // Copy the last element of the eScan buffer (which is the total size) to host
    }

    // Initialize permanent buffer objects
    glNamedBufferStorage(_buffers(BufferType::eNeighbors), _symmetricSize * sizeof(uint), nullptr, 0); // Each i's expanded neighbor set starts at eLayout[i].offset and contains eLayout[i].size neighbors, no longer including itself
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), _symmetricSize * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT); // Corresponding similarities
    glNamedBufferStorage(_buffers(BufferType::eDistances), _symmetricSize * sizeof(float), nullptr, 0); // Corresponding distances
    glAssert();

    // Update progress bar
    progressBar.setPostfix("Computing layout");
    progressBar.setProgress(4.0f / 6.0f);

    // 5.
    // Fill layout buffer
    {
      auto& timer = _timers(TimerType::eLayoutComp);
      timer.tick();

      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Symmetrizing similarities");
    progressBar.setProgress(5.0f / 6.0f);

    // 6.
    // Generate expanded similarities, neighbor and distances buffers, symmetrized and ready for use during the minimization
    {
      auto& timer = _timers(TimerType::eNeighborsComp);
      timer.tick();

      auto &program = _programs(ProgramType::eNeighborsComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbors", _params.k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffersTemp(BufferTempType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffersTemp(BufferTempType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffersTemp(BufferTempType::eCounts));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eDistances));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n * (_params.k-1), 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 7.
    // Sort neighbours within each KNN set (and corresponding similarities and distances)
    {
      auto &program = _programs(ProgramType::eNeighborsSortComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDistances));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert();
    }

    // Keep backup of similarities in eSimilaritiesOriginal, because eSimilarities may get changed
    glNamedBufferStorage(_buffers(BufferType::eSimilaritiesOriginal), _symmetricSize * sizeof(float), nullptr, 0);
    glCopyNamedBufferSubData(_buffers(BufferType::eSimilarities), _buffers(BufferType::eSimilaritiesOriginal), 0, 0, _symmetricSize * sizeof(float));

    // Update progress bar
    progressBar.setPostfix("Done!");
    progressBar.setProgress(1.0f);

    // Delete temporary buffers
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
    glAssert();

    // Output memory use of persistent OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::curt() << prefix << "Completed, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Poll twice so front/back timers are swapped
    glPollTimers(_timers.size(), _timers.data());
    glPollTimers(_timers.size(), _timers.data());
  }

  void Similarities::weighSimilarities(float weight, GLuint selectionBufferHandle, bool interOnly) {
    auto &program = _programs(ProgramType::eWeighSimilaritiesComp);
    program.bind();

    program.template uniform<uint>("nPoints", _params.n);
    program.template uniform<float>("weight", weight);
    program.template uniform<bool>("interOnly", interOnly);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNeighbors));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSimilarities));

    // Dispatch shader
    glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glAssert();
  }

  void Similarities::weighSimilaritiesPerAttribute(std::set<uint> weightedAttributeIndices, GLuint selectionBufferHandle, uint nSelected, GLuint labelsBufferHandle) {
    if(weightedAttributeIndices.size() == 0) { return; }
    
    // Create and initialize temp buffers. "...Sums" buffers aggregate link values per datapoint (i.e. each datapoint sums the values of its incident edges) 
    glCreateBuffers(_buffersTemp.size(), _buffersTemp.data());
    std::vector<uint> setvec(weightedAttributeIndices.begin(), weightedAttributeIndices.end());
    glNamedBufferStorage(_buffersTemp(BufferTempType::eWeightedAttributeIndices), weightedAttributeIndices.size() * sizeof(uint), setvec.data(), 0);

    // Weighting the similarities
    {
      auto &program = _programs(ProgramType::eWeighSimilaritiesPerAttributeComp);
      program.bind();

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("nHighDims", _params.nHighDims);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, selectionBufferHandle);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffersTemp(BufferTempType::eWeightedAttributeIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eDataset));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttributeWeights));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eSimilaritiesOriginal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eSimilarities));

      // Dispatch shader in batches of batchSize (selected) attriibutes
      uint batchSize = 10;
      for(uint b = 0; b * batchSize < weightedAttributeIndices.size(); ++b) {
        program.template uniform<uint>("batchBegin", b * batchSize);
        program.template uniform<uint>("batchEnd", std::min((b+1) * batchSize, (uint) weightedAttributeIndices.size()));
        glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
        glFinish();
        glAssert();
      }
    }

    // Renormalizing the similarities
    {
      float simSumOrg = reduce<float>(_buffers(BufferType::eSimilaritiesOriginal), true, selectionBufferHandle);
      float simSumNew = reduce<float>(_buffers(BufferType::eSimilarities), true, selectionBufferHandle);
      float factor = simSumOrg / simSumNew;
      weighSimilarities(factor, selectionBufferHandle);
    }

    //// DEBUGGING
    std::vector<uint> neig(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eNeighbors), 0, _symmetricSize * sizeof(uint), neig.data());
    std::vector<float> sims(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eSimilarities), 0, _symmetricSize * sizeof(float), sims.data());
    std::vector<float> simsO(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eSimilaritiesOriginal), 0, _symmetricSize * sizeof(float), simsO.data());
    std::vector<float> dist(_symmetricSize);
    glGetNamedBufferSubData(_buffers(BufferType::eDistances), 0, _symmetricSize * sizeof(float), dist.data());
    std::vector<uint> layo(_params.n * 2);
    glGetNamedBufferSubData(_buffers(BufferType::eLayout), 0, _params.n * 2 * sizeof(uint), layo.data());
    std::vector<uint> selc(_params.n);
    glGetNamedBufferSubData(selectionBufferHandle, 0, _params.n * sizeof(uint), selc.data());
    std::vector<int> labl(_params.n);
    glGetNamedBufferSubData(labelsBufferHandle, 0, _params.n * sizeof(int), labl.data());

    int classA = 1;
    int classB = 7;
    // Print inter-class and intra-class average similarity change
    std::vector<std::pair<uint, uint>> interNeighbs; std::vector<std::pair<uint, uint>> intraNeighbs;
    std::vector<float> interDeltas; std::vector<float> intraDeltas;
    std::vector<float> interDists; std::vector<float> intraDists;
    std::vector<float> interDistsAttr; std::vector<float> intraDistsAttr;
    std::vector<float> interDistsAttrRatios; std::vector<float> intraDistsAttrRatios;

    for(uint i = 0; i < _params.n; ++i) {
      if(selc[i] != 1 || (labl[i] != classA && labl[i] != classB)) { continue; }
      for(uint ij = layo[i*2+0]; ij < layo[i*2+0] + layo[i*2+1]; ++ij) {
        uint j = neig[ij];
        if(selc[j] != 1 || (labl[j] != classA && labl[j] != classB)) { continue; }

        float distAttrSum = 0.f;
        float distAttrRatioSum = 0.f;
        for (const uint &attr : weightedAttributeIndices) {
          float distAttr = std::pow(_dataPtr[i * _params.nHighDims + attr] - _dataPtr[j * _params.nHighDims + attr], 2);
          float distAttrRatio = std::sqrt(distAttr / dist[ij]);
          distAttrSum += distAttr;
          distAttrRatioSum += distAttrRatio;
        }

        float simDelta = sims[ij] / simsO[ij];
        if(labl[i] != labl[j]) {
          interNeighbs.emplace_back(i, j);
          interDeltas.push_back(simDelta);
          interDists.push_back(dist[ij]);
          interDistsAttr.push_back(distAttrSum);
          interDistsAttrRatios.push_back(distAttrRatioSum);
        }
        else {
          intraNeighbs.emplace_back(i, j);
          intraDeltas.push_back(simDelta);
          intraDists.push_back(dist[ij]);
          intraDistsAttr.push_back(distAttrSum);
          intraDistsAttrRatios.push_back(distAttrRatioSum);
        }
      }
    }
    std::cout << "\n\n";
    std::cout << "Delta inter: " << average(interDeltas) << "\n";
    std::cout << "Delta intra: " << average(intraDeltas) << "\n";
    std::cout << "Ratio inter: " << average(interDistsAttr) << " / " << average(interDists) << " = " << average(interDistsAttrRatios) << "\n";
    std::cout << "Ratio intra: " << average(intraDeltas) << " / " << average(intraDists) << " = " << average(intraDistsAttrRatios) << "\n";

    float sumSims = 0.f; float sumSimsPrev = 0.f;
    for(uint ij = 0; ij < sims.size(); ++ij) {
      sumSims += sims[ij]; sumSimsPrev += simsO[ij];
    }
    std::cout << "Total differences in similarities pre vs. post: " << sumSimsPrev << " - " << sumSims << " = " << sumSimsPrev - sumSims << "\n";
    // displayGraph(interDistsAttrRatios, intraDistsAttrRatios, false);
    glAssert();
    // writeBuffer<float>(_buffers(BufferType::eSimilarities), _symmetricSize, 1, "sims");
    // writeBuffer<float>(_buffers(BufferType::eSimilaritiesOriginal), _symmetricSize, 1, "simsO");
    glDeleteBuffers(_buffersTemp.size(), _buffersTemp.data());
  }

  void Similarities::reset() {
    glCopyNamedBufferSubData(_buffers(BufferType::eSimilaritiesOriginal), _buffers(BufferType::eSimilarities), 0, 0, _symmetricSize * sizeof(float));
  }

} // dh::sne