//
// Created by Kenji Nomura on 9/2/17.
//

#ifndef CPP_DEEPLEARNING_MNIST_IMAGE_H
#define CPP_DEEPLEARNING_MNIST_IMAGE_H

#include <cstdint>
#include <iostream>
#include <vector>
#include "IMnist.h"

class MnistImage : public IMnist {

 private:
  const std::string file_full_path;
  std::int32_t number_of_rows;
  std::int32_t number_of_cols;
  std::int32_t image_size;
  std::vector<std::vector<float>> images;

 public:
  MnistImage(const std::string &file_path);

  int getNumberOfData() const;

  /** size of image (rows * cols) */
  int size() const;

  int rows() const;
  int cols() const;

  std::vector<std::vector<float>> gets() const;
  std::vector<float> get(const int index) const;
  void show(const int number) const;

};

#endif //CPP_DEEPLEARNING_MNIST_IMAGE_H
