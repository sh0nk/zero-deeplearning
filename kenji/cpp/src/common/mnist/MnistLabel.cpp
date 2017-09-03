//
// Created by Kenji Nomura on 9/2/17.
//

#include "common/mnist/MnistLabel.h"
#include <fstream>

namespace {
constexpr int HEADER_SIZE = 8;
}

MnistLabel::MnistLabel(const std::string &file_path)
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

  std::vector<std::uint8_t> all_label(number_of_data);
  ifs.read((char *) all_label.data(), number_of_data);
  labels = all_label;
}

std::vector<std::uint8_t> MnistLabel::gets() const {
  return labels;
}

std::uint8_t MnistLabel::get(const int number) const {
  if (number >= number_of_data) {
    std::cerr << "No image of index " << number << ". 0-" << number_of_data - 1 << std::endl;
    return '\0';
  }
  return labels[number];
}

int MnistLabel::getNumberOfData() const {
  return number_of_data;
}

void MnistLabel::show(const int number) const {
  if (number >= number_of_data) {
    std::cerr << "No image of index " << number << ". 0-" << number_of_data - 1 << std::endl;
    return;
  }
  std::cout << static_cast<int>(labels[number]) << std::endl;
}
