//
// Created by Kenji Nomura on 8/19/17.
//

#ifndef CPP_DEEP_LEARNING_FUNCTIONS_H
#define CPP_DEEP_LEARNING_FUNCTIONS_H

#include <Eigen/Dense>

namespace functions {

/** Step Func (ステップ関数) */
Eigen::MatrixXd step_function(const Eigen::MatrixXd &X);

/** Sigmoid Func (シグモイド関数) */
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &X);

/** Softmax Func (ソフトマックス関数) */
Eigen::MatrixXd softmax(const Eigen::MatrixXd &X);

/** ReLU Func (ReLU関数) */
Eigen::MatrixXd relu(const Eigen::MatrixXd &X);

}

#endif //CPP_DEEP_LEARNING_FUNCTIONS_H
