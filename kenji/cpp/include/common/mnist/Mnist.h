//
// Created by Kenji Nomura on 9/3/17.
//

#ifndef CPP_DEEPLEARNING_MNIST_H
#define CPP_DEEPLEARNING_MNIST_H

#include "common/mnist/MnistImage.h"
#include "common/mnist/MnistLabel.h"

class Mnist {

 public:
  const MnistImage image;
  const MnistLabel label;

  Mnist(const std::string &image_file_path, const std::string &label_file_path);

  /** Number of data */
  int size() const;

};

#endif //CPP_DEEPLEARNING_MNIST_H
