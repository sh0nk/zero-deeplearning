//
// Created by Kenji Nomura on 9/1/17.
//

#include <iostream>
#include "catch.hpp"
#include "common/functions/detail/activation_functions.h"

TEST_CASE("functions#step_function", "[functions]") {
  Eigen::MatrixXd X1(2, 2);
  X1 << 1.0, 0.5,
      2.0, 1.5;
  CHECK(functions::activation::step_function(X1) == Eigen::MatrixXd::Ones(2, 2));

  Eigen::MatrixXd X2(2, 2);
  X2 << -1.0, -0.5,
      -2.0, -1.5;
  CHECK(functions::activation::step_function(X2) == Eigen::MatrixXd::Zero(2, 2));

  Eigen::MatrixXd X3(2, 2);
  X3 << 1.0, -0.5,
      -2.0, 1.5;
  CHECK(functions::activation::step_function(X3) == Eigen::MatrixXd::Identity(2, 2));
}

TEST_CASE("functions#sigmoid", "[functions]") {
  Eigen::MatrixXd A(2, 2);
  A << 1.0, 0.5,
      2.0, 1.5;
  Eigen::MatrixXd E(2, 2);
  E << 0.731059, 0.622459,
      0.880797, 0.817574;
  CHECK(functions::activation::sigmoid(A).isApprox(E, 1e-4));
}

TEST_CASE("functions#softmax", "[functions]") {
  Eigen::MatrixXd A(2, 2);
  A << 1.0, 0.5,
      2.0, 1.5;
  Eigen::MatrixXd E(2, 2);
  E << 0.167405, 0.101536,
      0.455054, 0.276004;
  CHECK(functions::activation::softmax(A).isApprox(E, 1e-4));
}

TEST_CASE("functions#relu", "[functions]") {
  Eigen::MatrixXd A(2, 2);
  A << -1.0, 0.5,
      2.0, -1.5;
  Eigen::MatrixXd E(2, 2);
  E << 0.0, 0.5,
      2.0, 0.0;
  CHECK(functions::activation::relu(A).isApprox(E, 1e-4));
}
