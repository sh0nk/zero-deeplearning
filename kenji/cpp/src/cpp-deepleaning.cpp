//
// Created by Kenji Nomura on 8/19/17.
//

#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include "Network.h"

int main() {
  Eigen::MatrixXd X(1, 2);
  X << 1.0, 0.5;

  Network network;
  network.loadDummyWeight();
  const Eigen::MatrixXd y = network.forward(X);
  std::cout << "y = " << y << std::endl; // 0.406259 0.593741

  return 0;
}
