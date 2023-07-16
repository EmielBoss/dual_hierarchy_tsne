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
  void readBinFile(std::string fileName,
                   std::vector<float>& data,
                   std::vector<int>& labels,
                   std::vector<float>& colors,
                   uint& n,
                   uint& d,
                   bool& noLabels,
                   int& nClasses)
  {
    bool includeAllClasses = nClasses == 0;

    // Clear vectors and create space to store data in
    data = std::vector<float>(n * d);
    labels = std::vector<int>(n, -1);
    colors = std::vector<float>(n * 3, 0.f);
    for(uint i = 0; i < n; ++i) { colors[i * 3] = 255.f; } // Make colors red by default

    std::ifstream ifsData(fileName + ".dat", std::ios::in | std::ios::binary);
    std::ifstream ifsLabels(fileName + ".lab", std::ios::in | std::ios::binary);
    std::ifstream ifsColors(fileName + ".col", std::ios::in | std::ios::binary);
    if(!ifsData) { throw std::runtime_error("Input data file cannot be accessed: " + fileName + ".dat"); }

    if(includeAllClasses) {
      ifsData.read((char *) data.data(), data.size() * sizeof(float));
      if(ifsLabels && !noLabels) {
        ifsLabels.read((char *) labels.data(), labels.size() * sizeof(float));
        std::set<int> classes(labels.begin(), labels.end());
        nClasses = classes.size();
        if(classes.find(0) == classes.end()) { // No 0 in classes means the first class is 1
          for (uint i = 0; i < n; ++i) { labels[i]--; }
        }
      }
      if(ifsColors) {
        ifsColors.read((char *) colors.data(), colors.size() * sizeof(float));
        std::cout << colors[0] << "," << colors[1] << std::endl;
      }
    } else {
      if(!ifsData) { throw std::runtime_error("Input label file cannot be accessed: " + fileName + ".dat (needed for --nClasses option)"); }
      int count = 0;
      for (uint i = 0; i < n; ++i) {
        int label = ifsLabels.peek();
        if(label < nClasses) {
          ifsData.read((char *) &data[d * count], d * sizeof(float));
          ifsLabels.read((char *) &labels[count], sizeof(int));
          if(ifsColors) { ifsColors.read((char *) &colors[count * 3], sizeof(float) * 3); }
          count++;
        } else {
          ifsData.ignore(d * sizeof(float));
          ifsLabels.ignore(sizeof(int));
          if(ifsColors) { ifsColors.ignore(sizeof(float) * 3); }
        }
      }
      labels.resize(count);
      data.resize(d * count);
      n = count;
    }
  }

  void readTxtClassNames(std::string fileName, std::vector<std::string>& classNames, int nClasses) {
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

  void writeBinFile(const std::string fileName,
                    const std::vector<float>& data,
                    const std::vector<int>& labels,
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
  
  void writeTextValuesFile(const std::string fileName, const std::vector<std::string>& values)
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