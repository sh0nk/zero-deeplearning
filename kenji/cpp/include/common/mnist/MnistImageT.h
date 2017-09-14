//
// Created by Kenji Nomura on 9/2/17.
//

#ifndef CPP_DEEPLEARNING_MNIST_IMAGE_T_H
#define CPP_DEEPLEARNING_MNIST_IMAGE_T_H

#include <cstdint>
#include <iostream>
#include <vector>
#include "IMnist.h"

#include <fstream>
#include <algorithm> // transform, copy_if

namespace {
constexpr int HEADER_SIZE = 16;

inline std::string getcwd_s() {
  char buffer[FILENAME_MAX];
  char *cwd = getcwd(buffer, FILENAME_MAX);
  if (cwd) return std::string(cwd);
  return "";
}

inline int byte4ToInt(const char *buffer, const int offset) {
  return int((unsigned char) (buffer[offset]) << 24 |
      (unsigned char) (buffer[offset + 1]) << 16 |
      (unsigned char) (buffer[offset + 2]) << 8 |
      (unsigned char) (buffer[offset + 3]));
}
}

template<typename T>
class MnistImageT {

 private:
  const std::string file_full_path;
  std::int32_t magic_number;
  std::int32_t number_of_data;
  std::int32_t number_of_rows;
  std::int32_t number_of_cols;
  std::int32_t image_size;
  std::vector<std::vector<std::uint8_t>> images;
  std::vector<std::vector<float>> images_normalized;
  bool is_normalized;

 public:
  MnistImageT(const std::string &file_path, const bool is_normalized)
      : file_full_path(getcwd_s() + "/../" + file_path), is_normalized(is_normalized) {
    std::ifstream ifs(file_full_path, std::ios::binary | std::ios::in);
    if (!ifs) {
      std::cerr << "failed in loading file: " << file_path << std::endl;
      return;
    }
    char headers[HEADER_SIZE];
    ifs.read(headers, HEADER_SIZE);
    // std::cout.write(headers, 16);
    // std::cout << std::endl;
    magic_number = byte4ToInt(headers, 0);
    number_of_data = byte4ToInt(headers, 4);
    number_of_rows = byte4ToInt(headers, 8);
    number_of_cols = byte4ToInt(headers, 12);
    image_size = number_of_rows * number_of_cols;

    for (int i = 0; i < number_of_data; i++) {
      std::vector<std::uint8_t> image(image_size);
      ifs.read((char *) image.data(), image_size);
      if (is_normalized) {
        std::vector<float> image_normalized(image_size);
        std::transform(image.begin(), image.end(), image_normalized.begin(), [](std::uint8_t pixel) { return pixel / 255.0f; });
        images_normalized.push_back(image_normalized);
      } else {
        images.push_back(image);
      }
      // std::cout.write(image, image_size);
      // std::cout << std::endl;
    }
  }

  // std::vector<std::uint8_t> get(const int number) const;
  std::vector<T> get(const int number) const {
    if (!is_normalized) throw std::invalid_argument("not normalized");
    if (number >= number_of_data) {
      std::cerr << "No image of index " << number << ". 0-" << number_of_data - 1 << std::endl;
      return {};
    }
    return images_normalized[number];
  }

};

#endif //CPP_DEEPLEARNING_MNIST_IMAGE_T_H
