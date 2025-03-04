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
#include <vector>
#include <set>
#include "dh/types.hpp"

namespace dh::util {
 /**
   * readBinFile(...)
   * 
   * Read a binary data file and interpret it as N D-dimensional vectors. Should the data file
   * contain labels for each vector, these can be read assuming they are stored as 32 bit uints.
   */
  void readBinFile(const std::string &fileName, 
                   std::vector<float> &data,
                   std::vector<int> &labels, 
                   uint n,
                   uint d,
                   bool withLabels,
                   int& nClasses,
                   bool includeAllClasses);

  /**
   * readTxtClassNames(...)
   * 
   * Read a text data file in which each line is a class name
   */
  void readTxtClassNames(const std::string &fileName,
                         std::vector<std::string>& classNames,
                         int nClasses);
  
  /**
   * writeBinFile(...)
   * 
   * Write a binary data file, storing N D-dimensional vectors. Should labels be available for
   * each vector, these can be interleaved with the data as 32 bit uints.
   */
  void writeBinFile(const std::string &fileName,
                    const std::vector<float> &data,
                    const std::vector<int> &labels,
                    uint n,
                    uint d,
                    bool withLabels = false);

  /**
   * writeTextValuesFile
   * 
   * Write a text file, storing specified string values. Useful for outputting time/kl divergence.
   */
  void writeTextValuesFile(const std::string &fileName,
                           const std::vector<std::string> &values);

  /**
   * readGLBuffer
   * 
   * Reads a text file in /dual_hierarchy_tsne/buffer_dumps/ into an OpenGL buffer. Useful for debugging. THE HANDLE IS RECREATED, so make sure to update any copies of it!
   */
  template<typename T>
  void readGLBuffer(GLuint& handle,
                    uint n,
                    uint d,
                    const std::string filename);
  
  /**
   * writeGLBuffer
   * 
   * Write an OpenGL buffer to a text file in /dual_hierarchy_tsne/buffer_dumps/ (create the folder beforehand). Useful for debugging
   */
  template<typename T>
  void writeGLBuffer(const GLuint handle,
                     uint n,
                     uint d,
                     const std::string filename);

  /**
   * readVector
   * 
   * Reads a text file in /dual_hierarchy_tsne/buffer_dumps/ into a vector. Useful for debugging
   */
  template<typename T>
  std::vector<T> readVector(uint n,
                            uint d,
                            const std::string filename);

  /**
   * writeVector
   * 
   * Write a vector to a text file in /dual_hierarchy_tsne/buffer_dumps/ (create the folder beforehand). Useful for debugging
   */
  template<typename T>
  void writeVector(const std::vector<T> vec,
                   uint n,
                   uint d,
                   const std::string filename);

  /**
   * readVector
   * 
   * Reads a text file in /dual_hierarchy_tsne/buffer_dumps/ into a vector. Useful for debugging
   */
  template<typename T>
  std::set<T> readSet(const std::string filename);

  /**
   * writeVector
   * 
   * Write a vector to a text file in /dual_hierarchy_tsne/buffer_dumps/ (create the folder beforehand). Useful for debugging
   */
  template<typename T>
  void writeSet(const std::set<T> vec, const std::string filename);
  
  /**
   * normalizeData
   * 
   * Normalize a vector between lower and upper (0 and 1 by default) with one global minimum and maximum
   */
  void normalizeData(std::vector<float>& data,
                     uint n,
                     uint d,
                     float lower = 0.f,
                     float upper = 1.f);

  /**
   * normalizeData
   * 
   * Normalize a vector between lower and upper (0 and 1 by default) with a separate minimum and maximum per dimension
   */
  void normalizeDataNonUniformDims(std::vector<float>& data,
                                   uint n,
                                   uint d,
                                   float lower = 0.f,
                                   float upper = 1.f);
} // dh::util