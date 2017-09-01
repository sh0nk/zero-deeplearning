//
// Created by Kenji Nomura on 8/22/17.
//

#ifndef CPP_DEEP_LEARNING_NETWORK_H
#define CPP_DEEP_LEARNING_NETWORK_H

#include <vector>
#include <Eigen/Dense>

class Network {

 private:
  std::vector<Eigen::MatrixXd> Ws;
  std::vector<Eigen::MatrixXd> Bs;

 public:
  Network();
  Network(const std::vector<Eigen::MatrixXd> &Ws, const std::vector<Eigen::MatrixXd> &Bs);

  Eigen::MatrixXd forward(const Eigen::MatrixXd &X) const;

  void setWeight(const std::vector<Eigen::MatrixXd> &Ws);
  void setBias(const std::vector<Eigen::MatrixXd> &Bs);

};

#endif //CPP_DEEP_LEARNING_NETWORK_H
