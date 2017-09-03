//
// Created by Kenji Nomura on 9/3/17.
//

#include "common/mnist/Mnist.h"

Mnist::Mnist(const std::string &image_file_path, const std::string &label_file_path)
    : image(image_file_path), label(label_file_path) {}

int Mnist::size() const {
  return image.getNumberOfData() == label.getNumberOfData() ? image.getNumberOfData() : -1;
}
