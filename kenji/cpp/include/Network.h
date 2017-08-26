//
// Created by Kenji Nomura on 8/22/17.
//

#ifndef CPP_DEEP_LEARNING_NETWORK_H
#define CPP_DEEP_LEARNING_NETWORK_H

#include <vector>
#include <Eigen/Core>

class Network {

 private:
  std::vector<Eigen::MatrixXd> W;
  std::vector<Eigen::VectorXd> b;

 public:
  void loadDummyWeight();
  Eigen::MatrixXd forward(const Eigen::MatrixXd &X);

};

inline void Network::loadDummyWeight() {
  Eigen::MatrixXd W1(2, 3);
  W1 << 0.1, 0.3, 0.5,
      0.2, 0.4, 0.6;
  W.push_back(W1);
  Eigen::VectorXd b1(3);
  b1 << 0.1, 0.2, 0.3;
  b.push_back(b1);

  Eigen::MatrixXd W2(3, 2);
  W2 << 0.1, 0.4,
      0.2, 0.5,
      0.3, 0.6;
  W.push_back(W2);
  Eigen::VectorXd b2(2);
  b2 << 0.1, 0.2;
  b.push_back(b2);

  Eigen::MatrixXd W3(2, 2);
  W3 << 0.1, 0.3,
      0.2, 0.4;
  W.push_back(W3);
  Eigen::VectorXd b3(2);
  b3 << 0.1, 0.2;
  b.push_back(b3);
}

#endif //CPP_DEEP_LEARNING_NETWORK_H
