//
// Created by Kenji Nomura on 8/19/17.
//

#include <iostream>
#include <thread>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include "common/functions/functions.h"
#include "common/utils.h"
#include "common/mnist/Mnist.h"

void step_function() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::activation::step_function(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-0.1, 1.1);
  plt::show();
}

void sigmoid() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::activation::sigmoid(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-0.1, 1.1);
  plt::show();
}

void relu() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::activation::relu(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-1.0, 5.5);
  plt::show();
}

int main() {
  const std::string image_path = "data/train-images-idx3-ubyte";
  const std::string label_path = "data/train-labels-idx1-ubyte";
  Mnist mnist(image_path, label_path);
  mnist.image.show(0);
  mnist.label.show(0);

  // step_function();
  // sigmoid();
  // relu();
  return 0;
}
