//
// Created by Kenji Nomura on 9/1/17.
//

#include <iostream>
#include "catch.hpp"
#include "Network.h"

void loadDummyWeight(Network &network);

TEST_CASE("Network#forward", "[network]") {
  Eigen::MatrixXd X(1, 2);
  X << 1.0, 0.5;

  Network network;
  loadDummyWeight(network);

  const Eigen::MatrixXd y = network.forward(X);
  std::cout << "y = " << y << std::endl; // 0.406259 0.593741

  Eigen::MatrixXd expected(1, 2);
  expected << 0.406259, 0.593741;

  CHECK(y.isApprox(expected, 1e-4));
}

void loadDummyWeight(Network &network) {
  std::vector<Eigen::MatrixXd> Ws;
  std::vector<Eigen::MatrixXd> Bs;

  Eigen::MatrixXd W1(2, 3);
  W1 << 0.1, 0.3, 0.5,
      0.2, 0.4, 0.6;
  Ws.push_back(W1);
  Eigen::MatrixXd b1(1, 3);
  b1 << 0.1, 0.2, 0.3;
  Bs.push_back(b1);

  Eigen::MatrixXd W2(3, 2);
  W2 << 0.1, 0.4,
      0.2, 0.5,
      0.3, 0.6;
  Ws.push_back(W2);
  Eigen::MatrixXd b2(1, 2);
  b2 << 0.1, 0.2;
  Bs.push_back(b2);

  Eigen::MatrixXd W3(2, 2);
  W3 << 0.1, 0.3,
      0.2, 0.4;
  Ws.push_back(W3);
  Eigen::MatrixXd b3(1, 2);
  b3 << 0.1, 0.2;
  Bs.push_back(b3);

  network.setWeight(Ws);
  network.setBias(Bs);
}
