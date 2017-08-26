//
// Created by Kenji Nomura on 8/19/17.
//

#ifndef CPP_DEEP_LEARNING_FUNCTIONS_H
#define CPP_DEEP_LEARNING_FUNCTIONS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace functions {

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &X) {
  return 1.0 / (1.0 + X.exp().array()).array();
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd &X) {
  const double c = X.maxCoeff();
  const Eigen::MatrixXd exp_A = (X.array() - c).exp();
  const double sum_exp_a = exp_A.sum();
  const Eigen::MatrixXd Y = exp_A.array() / sum_exp_a;
  return Y;
}

}

#endif //CPP_DEEP_LEARNING_FUNCTIONS_H
