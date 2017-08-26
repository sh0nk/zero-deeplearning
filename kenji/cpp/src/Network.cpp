//
// Created by Kenji Nomura on 8/22/17.
//

#include "Network.h"
#include <iostream>
#include "common/functions.h"

Eigen::MatrixXd Network::forward(const Eigen::MatrixXd &X) {
  std::cout << "X = " << X << std::endl;

  std::cout << X.cols() << ", " << W[0].rows() << std::endl;
  // auto A1 = X * W[0];
  auto A1 = X * W[0] + b[0];
  std::cout << "A1 = " << A1 << std::endl;

  auto Z1 = functions::sigmoid(A1);
  auto A2 = Z1 * W[1] + b[1];
  auto Z2 = functions::sigmoid(A2);
  auto A3 = Z2 * W[2] + b[2];
  auto y = functions::softmax(A3);
  return y;
}
