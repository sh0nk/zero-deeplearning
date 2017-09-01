//
// Created by Kenji Nomura on 8/19/17.
//

#include <iostream>
#include <thread>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include "common/functions.h"
#include "common/utils.h"

void step_function() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::step_function(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-0.1, 1.1);
  plt::show();
}

void sigmoid() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::sigmoid(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-0.1, 1.1);
  plt::show();
}

void relu() {
  const Eigen::MatrixXd A = utils::arange(-5.0, 5.0, 0.1);
  const std::vector<double> a = utils::convertEigenToStdVector(A);

  const Eigen::MatrixXd Y = functions::relu(A);
  const std::vector<double> y = utils::convertEigenToStdVector(Y);

  plt::plot(a, y);
  plt::ylim(-1.0, 5.5);
  plt::show();
}

int main() {
  step_function();
  sigmoid();
  relu();
  return 0;
}
