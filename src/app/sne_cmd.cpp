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

#include <exception>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cxxopts.hpp>
#include "dh/util/io.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/window.hpp"
#include "dh/vis/renderer.hpp"
#include "dh/sne/sne.hpp"

using uint = unsigned int;

// Constants
const std::string progDescr = "Demo application demonstrating dual-hierarchy t-SNE minimization.";
const std::string windowTitle = "Dual-Hierarchy t-SNE Demo";

// I/O and SNE parameters, set by cli(...)
std::string iptFilename;
std::string optFilename;
dh::sne::Params params;
std::vector<char> axisMapping(3, 't');

// Program parameters, set by cli(...)
bool progDoKlDivergence = false;
bool progDoLabels = false;
bool progDoVisDuring = false;
bool progDoVisAfter = false;

void cli(int argc, char** argv) {
  // Configure command line options
  cxxopts::Options options("sne_cmd", progDescr);
  options.add_options()
    // Required arguments
    ("iptFilename", "Input data file (required)", cxxopts::value<std::string>())
    ("nPoints", "number of data points (required)", cxxopts::value<uint>())
    ("nHighDims", "number of input dims (required)", cxxopts::value<uint>())
    ("nLowDims", "number of output dims (required)", cxxopts::value<uint>())
    
    // Optional parameter arguments
    ("p,perplexity", "Perplexity parameter (default: 30)", cxxopts::value<float>())
    ("i,iterations", "Number of minimization steps (default: 10000)", cxxopts::value<uint>())
    ("t,theta", "Approximation parameter (default: 0.25)", cxxopts::value<float>())

    // Optional program arguments
    ("o,optFilename", "Output data file (default: none)", cxxopts::value<std::string>())
    ("resWidth", "Window resolution width (default: 1536)", cxxopts::value<uint>())
    ("resHeight", "Window resolution height (default: 1024)", cxxopts::value<uint>())
    ("images", "Input data are images", cxxopts::value<bool>())
    ("imgWidth", "Image resolution width (default: 28)", cxxopts::value<uint>())
    ("imgHeight", "Image resolution height (default: 28)", cxxopts::value<uint>())
    ("imgDepth", "Image number of components (default: 1)", cxxopts::value<uint>())
    ("nClasses", "Use only the first n classes (default: number of distinct labels)", cxxopts::value<int>())
    ("nClusters", "An estimate of the number of t-SNE clusters, used for fixed datapoint force weighting", cxxopts::value<int>())
    ("lbl", "Input data file contains label data", cxxopts::value<bool>())
    ("kld", "Compute KL-Divergence", cxxopts::value<bool>())
    ("visDuring", "Visualize embedding during/after minimization", cxxopts::value<bool>())
    ("visAfter", "Visualize embedding after minimization", cxxopts::value<bool>())
    ("normalize", "Normalize data as preprocessing step", cxxopts::value<bool>())
    ("nonUniformDims", "Treat the dimensions/attributes as having different ranges and properties", cxxopts::value<bool>())
    ("disablePCA", "Disable PCA, which can be slow on some datasets", cxxopts::value<bool>())
    ("h,help", "Print this help message and exit")

    // Optional axis specifiers
    // ("x,xAxis", "What to map to the x-axis, t=t-SNE, p=PCA, a=Attribute, -=None (default: t)", cxxopts::value<char>());
    // ("y,yAxis", "What to map to the y-axis, t=t-SNE, p=PCA, a=Attribute, -=None (default: t)", cxxopts::value<char>());
    ("z,zAxis", "What to map to the z-axis, t=t-SNE, p=PCA, a=Attribute, -=None (default: t)", cxxopts::value<char>());

  options.parse_positional({"iptFilename", "nPoints", "nHighDims", "nLowDims"});
  options.positional_help("<iptFilename> <n> <nHighDims> <nLowDims>");
  auto result = options.parse(argc, argv);

  // Output help message as requested
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Check for required arguments, output help message otherwise
  if (!result.count("iptFilename") || !result.count("nPoints") 
      || !result.count("nHighDims") || !result.count("nLowDims")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Parse required arguments
  iptFilename = result["iptFilename"].as<std::string>();
  params.n = result["nPoints"].as<uint>();
  params.nHighDims = result["nHighDims"].as<uint>();
  params.nLowDims = result["nLowDims"].as<uint>();
  if(params.nLowDims == 2) { axisMapping[2] = '-'; }
  params.nPCs = std::min(params.nPCs, (int) params.nHighDims);

  // Check for and parse optional arguments
  if (result.count("optFilename")) { optFilename = result["optFilename"].as<std::string>(); }
  if (result.count("resWidth")) { params.resWidth = result["resWidth"].as<uint>(); }
  if (result.count("resHeight")) { params.resHeight = result["resHeight"].as<uint>(); }
  if (result.count("images")) { params.imageDataset = true; }
  if (result.count("imgWidth")) { params.imgWidth = result["imgWidth"].as<uint>(); }
  if (result.count("imgHeight")) { params.imgHeight = result["imgHeight"].as<uint>(); }
  if (result.count("imgDepth")) { params.imgDepth = result["imgDepth"].as<uint>(); }
  if (result.count("nClasses")) { params.nClasses = result["nClasses"].as<int>(); }
  if (result.count("nClusters")) { params.nClusters = result["nClusters"].as<int>(); }
  if (result.count("perplexity")) { params.perplexity = result["perplexity"].as<float>(); }
  if (result.count("iterations")) { params.iterations = result["iterations"].as<uint>(); }
  if (result.count("theta")) { params.dualHierarchyTheta = result["theta"].as<float>(); }
  // if (result.count("xAxis")) { axisMapping[0] = result["xAxis"].as<char>(); }
  // if (result.count("yAxis")) { axisMapping[1] = result["yAxis"].as<char>(); }
  if (result.count("zAxis")) { axisMapping[2] = result["zAxis"].as<char>(); }
  if (result.count("kld")) { progDoKlDivergence = true; }
  if (result.count("lbl")) { progDoLabels = true; }
  if (result.count("visDuring")) { progDoVisDuring = true; }
  if (result.count("visAfter")) { progDoVisAfter = true; }
  if (result.count("normalize")) { params.normalizeData = true; }
  if (result.count("nonUniformDims")) { params.uniformDims = false; }
  if (result.count("disablePCA")) { params.disablePCA = true; }
  params.datasetName = iptFilename.substr(0, iptFilename.length() - 4);
  params.nTexels = params.nHighDims / params.imgDepth;
}

void sne() {
  // Set up logger to use standard output stream for demo
  dh::util::Logger::init(&std::cout);

  // Load dataset
  std::vector<float> data;
  std::vector<int> labels;
  bool includeAllClasses = params.nClasses < 0;
  dh::util::readBinFile(iptFilename, data, labels, params.n, params.nHighDims, progDoLabels, params.nClasses, includeAllClasses);
  if(params.normalizeData) {
    if(params.uniformDims) { dh::util::normalizeData(data, params.n, params.nHighDims, 0.f, 255.f); }
    else { dh::util::normalizeDataNonUniformDims(data, params.n, params.nHighDims); }
  }
  if(!includeAllClasses) {
    params.n = data.size() / params.nHighDims;
    params.nClusters = params.nClasses;  
  }

  // Create OpenGL context (and accompanying invisible window)
  dh::util::GLWindowInfo info;
  {
    using namespace dh::util;
    info.title = windowTitle;
    info.width = params.resWidth;
    info.height = params.resHeight;
    info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                | GLWindowInfo::bOffscreen;
  }
  dh::util::GLWindow window(info);

  // Create necessary components
  dh::vis::Renderer renderer(&params, axisMapping.data(), window);
  dh::sne::SNE sne(&params, axisMapping, data, labels);

  // If visualization is requested, minimize and render at the same time
  if (progDoVisDuring) {
    sne.compSimilarities();

    // Spawn window, disabling vsync so the minimization is not locked to swap interval
    window.setVsync(false);
    window.setVisible(true);
    window.display();

    // Render loop, one minimization step between frames
    for (uint i = 0; i < params.iterations || params.iterateForever; ++i) {
      window.processEvents();
      sne.compMinimizationStep();
      renderer.render();
      window.display();
    }
  } else {
    sne.comp();
  }

  // Output timings
  dh::util::Logger::newl() << "Similarities runtime : " << sne.similaritiesTime();
  dh::util::Logger::newl() << "Minimization runtime : " << sne.minimizationTime();

  // If requested, output embedding to file 
  if (!optFilename.empty()) {
    dh::util::writeBinFile(optFilename, sne.embedding(), labels, params.n, params.nLowDims, progDoLabels);
  }

  // If requested, run visualization after minimization is completed
  if (progDoVisDuring || progDoVisAfter) {
    // Spawn window
    window.setVsync(true);
    window.setVisible(true);
    window.display();

    // Simple render loop
    while (window.canDisplay()) {
      window.processEvents();
      renderer.render();
      window.display();
    }
  }
}

int main(int argc, char** argv) {
  // Disable scientific notation when printing floats
  std::cout << std::fixed;
  std::cout << std::setprecision(10);

  try {
    cli(argc, argv);
    sne();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}