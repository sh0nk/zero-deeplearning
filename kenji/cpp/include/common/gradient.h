//
// Created by Kenji Nomura on 9/16/17.
//

#ifndef CPP_DEEPLEARNING_GRADIENT_H
#define CPP_DEEPLEARNING_GRADIENT_H

#include <Eigen/Dense>

namespace gradient {

double numerical_diff(const std::function<double(double)> &function, const double X) {
  constexpr double h = 1e-4; // 0.0001
  return (function(X + h) - function(X - h)) / (2 * h);
}

Eigen::VectorXd numerical_gradient(const std::function<double(Eigen::Vector2d)> &f, Eigen::VectorXd &X) {
  constexpr double h = 1e-4; // 0.0001
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(X.rows(), X.cols());

  for (int i = 0; i < X.rows(); i++) {
    double tmp_val = X(i);

    // calculate f(x+h)
    X(i) = tmp_val + h;
    double fxh1 = f(X);

    // calculate f(x-h)
    X(i) = tmp_val - h;
    double fxh2 = f(X);

    grad(i) = (fxh1 - fxh2) / (2 * h);
    X(i) = tmp_val; // revert
  }
  return grad;
}

Eigen::VectorXd gradient_descent(const std::function<double(Eigen::Vector2d)> &f, const Eigen::VectorXd &init_X, const double lr = 0.01, const int step_num = 100) {
  Eigen::VectorXd X = init_X;
  for (int i = 0; i < step_num; i++) {
    const Eigen::VectorXd grad = numerical_gradient(f, X);
    X = X - lr * grad;
  }
  return X;
}

}

#endif //CPP_DEEPLEARNING_GRADIENT_H
