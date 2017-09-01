//
// Created by Kenji Nomura on 9/1/17.
//

#ifndef CPP_DEEPLEARNING_UTILS_H
#define CPP_DEEPLEARNING_UTILS_H

namespace utils {

std::vector<double> convertEigenToStdVector(const Eigen::MatrixXd &X) {
  std::vector<double> x(X.rows() * X.cols());
  Eigen::Map<Eigen::MatrixXd>(&x[0], X.rows(), X.cols()) = X;
  return x;
}

Eigen::MatrixXd arange(const double start, const double stop, const double step) {
  const int size = (stop - start) / step;
  Eigen::MatrixXd X(1, size);

  for (int i = 0; i < size; i++) {
    X(0, i) = start + step * i;
  }

  return X;
}

}

#endif //CPP_DEEPLEARNING_UTILS_H
