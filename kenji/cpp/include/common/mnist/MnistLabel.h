//
// Created by Kenji Nomura on 9/2/17.
//

#ifndef CPP_DEEPLEARNING_MNIST_LABEL_H
#define CPP_DEEPLEARNING_MNIST_LABEL_H

#include <cstdint>
#include <iostream>
#include <vector>
#include "IMnist.h"

class MnistLabel : public IMnist {

 private:
  const std::string file_full_path;
  std::vector<std::uint8_t> labels; // value range 0-9

 public:
  MnistLabel(const std::string &file_path);

  int getNumberOfData() const;
  void show(const int number) const;
  std::vector<std::uint8_t> gets() const;
  std::uint8_t get(const int index) const;

};

#endif //CPP_DEEPLEARNING_MNIST_LABEL_H
