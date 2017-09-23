//
// Created by Kenji Nomura on 9/1/17.
//

#include "common/functions/detail/activation_functions.h"
#include <unsupported/Eigen/MatrixFunctions>

namespace functions {
namespace activation {

Eigen::MatrixXd step_function(const Eigen::MatrixXd &X) {
  return (X.array() > 0.0).cast<double>();
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &X) {
  return 1.0 / (1.0 + (X.array() * -1.0).exp());
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd &X) {
  const double c = X.maxCoeff();
  const Eigen::MatrixXd exp_A = (X.array() - c).exp();
  const double sum_exp_a = exp_A.sum(); // FIXME: 行毎
  const Eigen::MatrixXd Y = exp_A.array() / sum_exp_a;
  return Y;
}

Eigen::MatrixXd relu(const Eigen::MatrixXd &X) {
  return (X.array() > 0).select(X, 0); // (X > 0 ? X : 0)
  // same as above
  // return X.array().max(Eigen::MatrixXd::Zero(X.rows(), X.cols()).array());
}

}
}
