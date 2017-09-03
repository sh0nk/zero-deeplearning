//
// Created by Kenji Nomura on 9/2/17.
//

#include "common/mnist/MnistImage.h"
#include <fstream>
#include <algorithm> // transform, copy_if

namespace {
constexpr int HEADER_SIZE = 16;
}

MnistImage::MnistImage(const std::string &file_path)
    : file_full_path(getcwd_s() + "/../" + file_path) {
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
    std::vector<float> image_normalized(image_size);
    std::transform(image.begin(), image.end(), image_normalized.begin(), [](std::uint8_t pixel) { return pixel / 255.0f; });
    images.push_back(image_normalized);
    // std::cout.write(image, image_size);
    // std::cout << std::endl;
  }
}

std::vector<std::vector<float>> MnistImage::gets() const {
  return images;
}

std::vector<float> MnistImage::get(const int index) const {
  if (index >= number_of_data) {
    std::cerr << "No image of index " << index << ". 0-" << number_of_data - 1 << std::endl;
    return {};
  }
  return images[index];
}

int MnistImage::getNumberOfData() const {
  return number_of_data;
}

int MnistImage::size() const {
  return number_of_rows * number_of_cols;
}

int MnistImage::rows() const {
  return number_of_rows;
}

int MnistImage::cols() const {
  return number_of_cols;
}

void MnistImage::show(const int number) const {
  if (number >= number_of_data) {
    std::cerr << "No image of index " << number << ". 0-" << number_of_data - 1 << std::endl;
    return;
  }
  for (int row = 0; row < number_of_rows; row++) {
    for (int col = 0; col < number_of_cols; col++) {
      const float pixel = images[number][row * number_of_rows + col];
      // const std::uint8_t pixel = images[number][row * number_of_rows + col];
      // std::cout << int(pixel) << ",";
      // std::cout << (pixel > 0 ? "1" : "0") << ",";
      std::cout << (pixel > 0.0f ? "■" : "□");
    }
    std::cout << std::endl;
  }
}
