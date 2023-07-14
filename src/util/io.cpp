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

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <set>
#include "dh/util/io.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::util {
  void readBinFile(const std::string &fileName,
                   std::vector<float> &data,
                   std::vector<int> &labels,
                   uint n,
                   uint d,
                   bool withLabels,
                   int& nClasses,
                   bool includeAllClasses)
  {
    std::ifstream ifs(fileName, std::ios::in | std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Input file cannot be accessed: " + fileName);
    }

    // Clear vectors and create space to store data in
    data = std::vector<float>(n * d);
    labels = std::vector<int>(n, -1);

    // Read data, either in a single call (no labels) or intermittently (to extract labels)
    if (withLabels) {
      if(includeAllClasses) {
        std::set<int> classes;
        for (uint i = 0; i < n; ++i) {
          ifs.read((char *) &labels[i], sizeof(int));
          ifs.read((char *) &data[d * i], d * sizeof(float));
          classes.insert(labels[i]);
        }
        nClasses = classes.size();
        if(classes.find(0) == classes.end()) { // No 0 in classes means the first class is 1
          for (uint i = 0; i < n; ++i) { labels[i]--; }
        }
      } else {
        int count = 0;
        for (uint i = 0; i < n; ++i) {
          int label = ifs.peek();
          if(label < nClasses) {
            ifs.read((char *) &labels[count], sizeof(int));
            ifs.read((char *) &data[d * count], d * sizeof(float));
            count++;
          } else {
            ifs.ignore(sizeof(int) + d * sizeof(float));
          }
        }
        labels.resize(count);
        data.resize(d * count);
      }
    } else {
      ifs.read((char *) data.data(), data.size() * sizeof(float));
    }
  }

  void readTxtClassNames(const std::string &fileName, std::vector<std::string>& classNames, int nClasses) {
    std::ifstream file(fileName);
    if (!file) {
      return;
    }
    uint counter = 0;
    std::string str;
    while (std::getline(file, str))
    {
        classNames[counter++] = str;
        if(counter > (uint) nClasses) { return; }
    }
  }

  void writeBinFile(const std::string &fileName,
                    const std::vector<float> &data,
                    const std::vector<int> &labels,
                    uint n,
                    uint d,
                    bool withLabels)
  {
    std::ofstream ofs(fileName, std::ios::out | std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    if (withLabels) {
      for (uint i = 0; i < n; ++i) {
        ofs.write((char *) &labels[i], sizeof(int));
        ofs.write((char *) &data[d * i], d * sizeof(float));
      }
    } else {
      ofs.write((char *) data.data(), data.size() * sizeof(float));
    }
  }
  
  void writeTextValuesFile(const std::string &fileName, const std::vector<std::string> &values)
  {
    std::ofstream ofs (fileName, std::ios::out);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    for (const auto &v : values) {
      ofs << v << '\n';
    }
  }

  template<typename T>
  void readGLBuffer(GLuint& handle, const std::string filename) {
    std::vector<T> buffer = readVector<T>(filename);
    GLint flags;
    glGetNamedBufferParameteriv(handle, GL_BUFFER_STORAGE_FLAGS, &flags);
    glDeleteBuffers(1, &handle);
    glCreateBuffers(1, &handle);
    glNamedBufferStorage(handle, buffer.size() * sizeof(T), buffer.data(), flags);
  }

  template<typename T>
  void writeGLBuffer(const GLuint handle, uint n, uint d, const std::string filename) {
    std::vector<T> buffer(n * d);
    glGetNamedBufferSubData(handle, 0, n * d * sizeof(T), buffer.data());
    glAssert();
    writeVector<T>(buffer, n, d, filename);
  }

  template<typename T>
  std::vector<T> readVector(const std::string filename) {
    std::vector<T> vec;
    std::ifstream file("./buffer_dumps/" + filename + ".txt");
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string val_str;
      while(std::getline(iss, val_str, '|')) {
        std::istringstream val_iss(val_str);
        T val;
        val_iss >> val;
        vec.push_back(val);
      }
    }
    
    return vec;
  }

  template<typename T>
  void writeVector(const std::vector<T> vec, uint n, uint d, const std::string filename) {
    std::ofstream file("./buffer_dumps/" + filename + ".txt");
    for(uint i = 0; i < n; i++) {
      for(uint j = 0; j < d; ++j) {
        T val = vec[i * d + j];
        file << val << "|";
      }
      file << "\n";
    }
  }

  template<typename T>
  std::set<T> readSet(const std::string filename) {
    std::set<T> set;
    std::ifstream file("./buffer_dumps/" + filename + ".txt");
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return set;
    }
    std::string str;
    while (std::getline(file, str)) {
      set.insert(std::stoi(str));
    }
    
    return set;
  }

  template<typename T>
  void writeSet(const std::set<T> set, const std::string filename) {
    std::ofstream file("./buffer_dumps/" + filename + ".txt");
    for (const auto& str : set) {
      file << str << "\n";
    }
  }

  void normalizeData(std::vector<float>& data, uint n, uint d, float lower, float upper) {
    // Determine min and max attribute value
    auto [minIt, maxIt] = std::minmax_element(data.begin(), data.end());
    float min = *minIt; float max = *maxIt;

    for(uint i = 0; i < n * d; ++i) {
      data[i] = (data[i] - min) / (max - min);
      data[i] = data[i] * (upper - lower) + lower;
      if(data[i] != data[i]) { data[i] = 0.f; }
    }
  }

  void normalizeDataNonUniformDims(std::vector<float>& data, uint n, uint d, float lower, float upper) {
    // Determine min and max attribute values per attribute
    std::vector<float> mins(d,  FLT_MAX);
    std::vector<float> maxs(d, -FLT_MAX);
    for(uint i = 0; i < n; ++i) {
      for(uint a = 0; a < d; ++a) {
        if(data[i * d + a] < mins[a]) { mins[a] = data[i * d + a]; }
        if(data[i * d + a] > maxs[a]) { maxs[a] = data[i * d + a]; }
      }
    }

    for(uint i = 0; i < n; ++i) {
      for(uint a = 0; a < d; ++a) {
        data[i * d + a] = (data[i * d + a] - mins[a]) / (maxs[a] - mins[a]);
        data[i * d + a] = data[i * d + a] * (upper - lower) + lower;
        if(data[i * d + a] != data[i * d + a]) { data[i * d + a] = 0.f; }
      }
    }
  }

  // Template instantiations for writeGLBuffer for float, int, uint
  template void readGLBuffer<float>(GLuint& handle, const std::string filename);
  template void readGLBuffer<uint>(GLuint& handle, const std::string filename);
  template void readGLBuffer<int>(GLuint& handle, const std::string filename);

  template void writeGLBuffer<float>(const GLuint handle, uint n, uint d, const std::string filename);
  template void writeGLBuffer<uint>(const GLuint handle, uint n, uint d, const std::string filename);
  template void writeGLBuffer<int>(const GLuint handle, uint n, uint d, const std::string filename);

  template std::set<uint> readSet<uint>(const std::string filename);
  template void writeSet<uint>(const std::set<uint> set, const std::string filename);
} // dh::util