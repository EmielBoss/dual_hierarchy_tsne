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
#include "dh/types.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/gl/program.hpp"

namespace dh::util {
  enum class BufferReduceType {
    eReduce,
    eReduced,
    eAccumulationPerDatapoint,

    Length
  };

  enum class BufferRemoveType {
    eCumSum,
    eRemoved,

    Length
  };

  enum class ProgramType {
    eAccumulatePerDatapointFloatComp,
    eAccumulatePerDatapointUintComp,
    eReduceFloatComp,
    eReduceUintComp,

    eRemoveFloatComp,
    eRemoveUintComp,

    Length,
  };

  class Reducer {
    public:
      // Accessor; there is one Reducer used by the util library
      // Ergo, InputQueue implements a singleton pattern, but
      // with controllable initialization/destruction. It is not 
      // functional for the sne lib until Reducer::instance().init() has been called.
      static Reducer& instance() {
        static Reducer instance;
        return instance;
      }

      // Setup/teardown functions
      void init();
      void dstr();

      // Sum reduction on a buffer
      // largeBuffer means larger than _params->n, in which case its contents are first accumulated per (selected) datapoint
      // If largeBuffer, layoutBuffer and neighborBuffer need to be > 0 as well
      // If countVal > 0, instead of summing all (selected) values, the number of occurences of countVal are counted
      template <typename T> T reduce(GLuint& bufferToReduce, uint n, T countVal = 0, bool largeBuffer = false, GLuint selectionBuffer = 0, GLuint layoutBuffer = 0, GLuint neighborsBuffer = 0);
      template <typename T> uint remove(GLuint& bufferToRemove, uint n, uint d, GLuint selectionBuffer);

      bool isInit() const { return _isInit; }

    private:
      // Hidden constr/destr
      Reducer();
      ~Reducer();

      // State
      bool _isInit;

      // Objects
      util::EnumArray<ProgramType, util::GLProgram> _programs;
      util::EnumArray<BufferReduceType, GLuint> _buffersReduce;
      util::EnumArray<BufferRemoveType, GLuint> _buffersRemove;
  };
} // dh::util