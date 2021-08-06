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


#include <vector>
#include "types.hpp"
#include "util/logger.hpp"
#include "sne/sne_params.hpp"
#include "sne/sne_similarities.hpp"
#include "sne/sne_minimization.hpp"

namespace dh::sne {
  template <uint D> // Dimension of produced embedding
  class SNE {
  public:
    // Constr/destr
    SNE();
    SNE(const std::vector<float>& data, SNEParams params, util::Logger* logger = nullptr);
    ~SNE();

    // Copy constr/assignment is explicitly deleted (no copying underlying handles)
    SNE(const SNE&) = delete;
    SNE& operator=(const SNE&) = delete;

    // Move constr/operator moves handles
    SNE(SNE&&) noexcept;
    SNE& operator=(SNE&&) noexcept;

    // Compute similarities, then perform the minimization
    void comp();

    // Compute high-dimensional similarities only
    void compSimilarities();

    void prepMinimization();

    // Perform a single step of the minimization only
    void compIteration();

    // Perform all steps of the minimization only
    void compMinimization();
    
    // Compute and return KL-Divergence of current embedding
    // Don't do this while minimizing unless you don't care about performance
    float klDivergence() const;

    // Return raw data of current embedding
    // Don't do this while minimizing unless you don't care about performance
    std::vector<float> embedding() const;

    // Returns state of components
    bool isInit() const { return _isInit; }

  private:
    // State
    bool _isInit;
    uint _iteration;
    SNEParams _params;
    util::Logger* _logger;

    // Subcomponents
    SNESimilarities<D> _sneSimilarities;
    SNEMinimization<D> _sneMinimization;

  public:
    // Swap internals with another object
    friend void swap(SNE& a, SNE& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._iteration, b._iteration);
      swap(a._params, b._params);
      swap(a._sneSimilarities, b._sneSimilarities);
      swap(a._sneMinimization, b._sneMinimization);
    }
  };
} // dh::sne