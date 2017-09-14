//
// Created by Kenji Nomura on 9/4/17.
//

#ifndef CPP_DEEPLEARNING_LOST_FUNCTIONS_H
#define CPP_DEEPLEARNING_LOST_FUNCTIONS_H

#include <Eigen/Dense>

namespace functions {
namespace lost {

double mean_squared_error(const Eigen::RowVectorXd &Y, const Eigen::RowVectorXd &T);

double cross_entropy_error(const Eigen::RowVectorXd &Y, const Eigen::RowVectorXd &T);

Eigen::MatrixXd cross_entropy_error(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &T);

}
}

#endif //CPP_DEEPLEARNING_LOST_FUNCTIONS_H
