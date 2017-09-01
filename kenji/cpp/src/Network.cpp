//
// Created by Kenji Nomura on 8/22/17.
//

#include "Network.h"
#include <iostream>
#include "common/functions.h"

Eigen::MatrixXd Network::forward(const Eigen::MatrixXd &X) {
  std::cout << "X = " << X << std::endl;

  Eigen::MatrixXd Ai;
  Eigen::MatrixXd Zi = X;
  for (size_t i = 0; i < Ws.size(); i++) {
    Ai = Zi * Ws[i] + Bs[i];
    std::cout << "A" << i << " = " << Ai << std::endl;

    if (i == Ws.size() - 1L) break;
    Zi = functions::sigmoid(Ai);
    std::cout << "Z" << i << " = " << Zi << std::endl;
  }

  const auto y = functions::softmax(Ai);
  return y;

  /*
  auto A1 = X * W[0] + b[0];
  std::cout << "A1 = " << A1 << std::endl;

  auto Z1 = functions::sigmoid(A1);
  std::cout << "Z1 = " << Z1 << std::endl;

  auto A2 = Z1 * W[1] + b[1];
  std::cout << "A2 = " << A2 << std::endl;

  auto Z2 = functions::sigmoid(A2);
  std::cout << "Z2 = " << Z2 << std::endl;

  auto A3 = Z2 * W[2] + b[2];
  std::cout << "A3 = " << A3 << std::endl;

  auto y = functions::softmax(A3);
  return y;
  */
}
