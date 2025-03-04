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

#include <string>
#include "dh/types.hpp"

namespace dh::sne {
  struct Params {
    // Input dataset params
    uint n = 0;
    uint nHighDims = 0;
    uint nLowDims = 2; // 2 or 3 only
    int nClusters = 10; // If unsure, leave this at the default 10
    int nClasses = -1; // -1 indicates all classes
    uint nTexels; // Number of texels of images in case imageDataset
    bool normalizeData = false;
    bool uniformDims = true;
    std::string datasetName = "";

    // Basic tSNE parameters
    uint iterations = 10000;
    bool iterateForever = true;
    float perplexity = 30.f;
    uint kMax = 192; // Don't exceeed this value for big vector datasets unless you have a lot of coffee and memory
    uint k = std::min(kMax, 3 * (uint)(perplexity) + 1);

    // Approximation parameters
    float singleHierarchyTheta = 0.5f;
    float dualHierarchyTheta = 0.25f;
    float fieldScaling2D = 2.0f;
    float fieldScaling3D = 1.2f;

    // Embedding initialization parameters
    int seed = 1;
    float rngRange = 0.1f;
    
    // Gradient descent iteration parameters
    uint momentumSwitchIter = 250;
    uint nExaggerationIters = 250;
    uint nExponentialDecayIters = 150;

    // Gradient descent parameters
    float minimumGain = 0.1f;
    float eta = 200.f;

    // Original params
    float momentum = 0.2f;
    float finalMomentum = 0.5f;
    float exaggerationFactor = 4.0f;

    // Params used by TSNE-CUDA, much better on some datasets such as ImageNet
    // float momentum = 0.5f;
    // float finalMomentum = 0.8f;
    // float exaggerationFactor = 12.0f;

    // Program params
    uint resWidth = 1920;
    uint resHeight = 920;
    int nPCs = 10;
    bool disablePCA = false;
    float maxAttributeWeight = 2.f;
    float maxSimilarityWeight = 3.f;

    // Image dataset params
    bool imageDataset = false;
    uint imgWidth = 28;
    uint imgHeight = 28;
    uint imgDepth = 1;
  };
} // dh::sne
