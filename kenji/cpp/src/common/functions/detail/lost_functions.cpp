//
// Created by Kenji Nomura on 9/4/17.
//

#include "common/functions/detail/lost_functions.h"
#include <iostream>

namespace functions {
namespace lost {

double mean_squared_error(const Eigen::RowVectorXd &Y, const Eigen::RowVectorXd &T) {
  return 0.5 * (Y.array() - T.array()).pow(2.0).sum();
}

double cross_entropy_error(const Eigen::RowVectorXd &Y, const Eigen::RowVectorXd &T) {
  constexpr double delta = 1e-7;
  return -1.0 * (T.array() * (Y.array() + delta).log()).sum();
}

Eigen::MatrixXd cross_entropy_error(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &T) {
  const int batch_size = Y.rows();
  // return -1.0 * (T.array() * (Y.array() + delta).log()).sum() / batch_size;
}

}
}
