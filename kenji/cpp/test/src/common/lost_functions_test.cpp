//
// Created by Kenji Nomura on 9/4/17.
//

#include <iostream>
#include "catch.hpp"
#include "common/functions/detail/lost_functions.h"

TEST_CASE("lost_functions#mean_squared_error", "[functions]") {
  Eigen::RowVectorXd Y1(2);
  Y1 << 1.0, 0.5;
  Eigen::RowVectorXd T1(2);
  T1 << 1.0, 0.0;
  CHECK(functions::lost::mean_squared_error(Y1, T1) == 0.125);

  Eigen::RowVectorXd Y2(2);
  Y2 << 1.0, -0.5;
  CHECK(functions::lost::mean_squared_error(Y2, T1) == 0.125);
}

TEST_CASE("lost_functions#mean_squared_error 2", "[functions]") {
  Eigen::RowVectorXd Y1(10);
  Y1 << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;
  Eigen::RowVectorXd T1(10);
  T1 << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
  CHECK(functions::lost::mean_squared_error(Y1, T1) == Approx(0.097500000000000031));

  Eigen::RowVectorXd Y2(10);
  Y2 << 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0;
  CHECK(functions::lost::mean_squared_error(Y2, T1) == Approx(0.59750000000000003));
}

TEST_CASE("lost_functions#cross_entropy_error", "[functions]") {
  Eigen::RowVectorXd Y1(10);
  Y1 << 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0;
  Eigen::RowVectorXd T1(1, 10);
  T1 << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
  CHECK(functions::lost::cross_entropy_error(Y1, T1) == Approx(0.51082545709933802));

  Eigen::RowVectorXd Y2(1, 10);
  Y2 << 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0;
  CHECK(functions::lost::cross_entropy_error(Y2, T1) == Approx(2.3025840929945458));
}
