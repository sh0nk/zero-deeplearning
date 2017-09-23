//
// Created by Kenji Nomura on 9/1/17.
//

#ifndef CPP_DEEPLEARNING_UTILS_H
#define CPP_DEEPLEARNING_UTILS_H

namespace utils {

/**
 * Convert Eigen Matrix to std::vector.
 */
std::vector<double> convertEigenToStdVector(const Eigen::MatrixXd &X);

/**
 * Return evenly spaced values within a given interval. Migrate from `numpy.arange`.
 * @param start Start of interval.
 * @param stop End of interval.
 * @param step Spacing between values.
 * @return Array of evenly spaced values.
 */
Eigen::MatrixXd arange(const double start, const double stop, const double step);

}

inline Eigen::MatrixXd utils::arange(const double start, const double stop, const double step) {
  const int size = (stop - start) / step;
  Eigen::MatrixXd X(1, size);

  for (int i = 0; i < size; i++) {
    X(0, i) = start + step * i;
  }

  return X;
}

inline std::vector<double> utils::convertEigenToStdVector(const Eigen::MatrixXd &X) {
  std::vector<double> x(X.rows() * X.cols());
  Eigen::Map<Eigen::MatrixXd>(&x[0], X.rows(), X.cols()) = X;
  return x;
}

#endif //CPP_DEEPLEARNING_UTILS_H
