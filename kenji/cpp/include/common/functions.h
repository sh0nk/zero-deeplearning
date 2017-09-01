//
// Created by Kenji Nomura on 8/19/17.
//

#ifndef CPP_DEEP_LEARNING_FUNCTIONS_H
#define CPP_DEEP_LEARNING_FUNCTIONS_H

#include <Eigen/Dense>

namespace functions {

/** ステップ関数 */
Eigen::MatrixXd step_function(const Eigen::MatrixXd &X);

/** シグモイド関数 */
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &X);

/** ソフトマックス */
Eigen::MatrixXd softmax(const Eigen::MatrixXd &X);

}

#endif //CPP_DEEP_LEARNING_FUNCTIONS_H
