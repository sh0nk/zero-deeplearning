//
// Created by Kenji Nomura on 9/16/17.
//

#include <iostream>
#include "catch.hpp"
#include "common/gradient.h"
#include "common/functions/functions.h"
using namespace Eigen;

namespace {
/// y = 0.01 x^2 + 0.1 x
double function_1(const double x) {
  return 0.01 * x * x + 0.1 * x;
}
/// f(x0, x1) = x0^2 + x1^2
double function_2(const Eigen::Vector2d &X) {
  return X(0) * X(0) + X(1) * X(1);
}
}

/// 4.3.2 数値微分の例
TEST_CASE("gradient#numerical_diff", "[functions]") {
  // f(x) = 0.01x^2 + 0.1x
  // df(x)/dx = 0.02x + 0.1
  std::function<double(double)> f = [](double x) -> double { return function_1(x); };

  double ret1 = gradient::numerical_diff(f, 5.0);
  std::cout << "numerical_diff 1: " << ret1 << std::endl;
  CHECK(ret1 == Approx(0.2));

  double ret2 = gradient::numerical_diff(f, 10.0);
  std::cout << "numerical_diff 2: " << ret2 << std::endl;
  CHECK(ret2 == Approx(0.3));
}

/// 4.3.3 偏微分
TEST_CASE("gradient#numerical_gradient", "[functions]") {
  // f(x0, x1) = x0^2 + x1^2
  std::function<double(Eigen::Vector2d)> f = [](Eigen::Vector2d X) -> double { return function_2(X); };

  Eigen::VectorXd X1(2);
  X1 << 3.0, 4.0;
  Eigen::VectorXd ret1 = gradient::numerical_gradient(f, X1);
  std::cout << "numerical_gradient 1: " << ret1.transpose() << std::endl;
  CHECK(ret1.isApprox(Eigen::Vector2d(6.0, 8.0), 1e-4));

  Eigen::VectorXd X2(2);
  X2 << 0.0, 2.0;
  Eigen::VectorXd ret2 = gradient::numerical_gradient(f, X2);
  std::cout << "numerical_gradient 2: " << ret2.transpose() << std::endl;
  CHECK(ret2.isApprox(Eigen::Vector2d(0.0, 4.0), 1e-4));

  Eigen::VectorXd X3(2);
  X3 << 3.0, 0.0;
  Eigen::VectorXd ret3 = gradient::numerical_gradient(f, X3);
  std::cout << "numerical_gradient 3: " << ret3.transpose() << std::endl;
  CHECK(ret3.isApprox(Eigen::Vector2d(6.0, 0.0), 1e-4));
}

/// 4.4.1 勾配法
TEST_CASE("gradient#gradient_descent", "[functions]") {
  // f(x0, x1) = x0^2 + x1^2
  std::function<double(Eigen::Vector2d)> f = [](Eigen::Vector2d X) -> double { return function_2(X); };

  Eigen::VectorXd init_X1(2);
  init_X1 << -3.0, 4.0;
  Eigen::VectorXd ret1 = gradient::gradient_descent(f, init_X1, 0.1, 100);
  std::cout << "gradient_descent 1: " << ret1.transpose() << std::endl;
  CHECK(ret1.isApprox(Eigen::Vector2d(-6.11110793e-10, 8.14814391e-10), 1e-4)); // almost (0, 0)

  // 学習率が大きすぎる例: lr=10.0
  Eigen::VectorXd ret2 = gradient::gradient_descent(f, init_X1, 10.0, 100);
  std::cout << "gradient_descent 2: " << ret2.transpose() << std::endl;
  CHECK(ret2.isApprox(Eigen::Vector2d(-2.58983747e+13, -1.29524862e+12), 1e-4));

  // 学習率が小さすぎる例: lr=1e-10
  Eigen::VectorXd ret3 = gradient::gradient_descent(f, init_X1, 1e-10, 100);
  std::cout << "gradient_descent 3: " << ret3.transpose() << std::endl;
  CHECK(ret3.isApprox(Eigen::Vector2d(-2.99999994, 3.99999992), 1e-4));
}
