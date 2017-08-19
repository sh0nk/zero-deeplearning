//
// Created by Kenji Nomura on 8/19/17.
//

#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

int main(int argc, char *argv[]) {
  const double pi = std::acos(-1.0);
  Eigen::MatrixXd A(3, 3);
  A << 0, -pi / 4, 0,
      pi / 4, 0, 0,
      0, 0, 0;
  std::cout << "test" << std::endl;
  std::cout << "The matrix A is:\n" << A << "\n\n";
  std::cout << "The matrix exponential of A is:\n" << A.exp() << "\n\n";
  return 0;
}
