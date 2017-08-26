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
  void loadDummyWeight();
  Eigen::MatrixXd forward(const Eigen::MatrixXd &X);

};

inline void Network::loadDummyWeight() {
  Eigen::MatrixXd W1(2, 3);
  W1 << 0.1, 0.3, 0.5,
      0.2, 0.4, 0.6;
  Ws.push_back(W1);
  Eigen::MatrixXd b1(1, 3);
  b1 << 0.1, 0.2, 0.3;
  Bs.push_back(b1);

  Eigen::MatrixXd W2(3, 2);
  W2 << 0.1, 0.4,
      0.2, 0.5,
      0.3, 0.6;
  Ws.push_back(W2);
  Eigen::MatrixXd b2(1, 2);
  b2 << 0.1, 0.2;
  Bs.push_back(b2);

  Eigen::MatrixXd W3(2, 2);
  W3 << 0.1, 0.3,
      0.2, 0.4;
  Ws.push_back(W3);
  Eigen::MatrixXd b3(1, 2);
  b3 << 0.1, 0.2;
  Bs.push_back(b3);
}

#endif //CPP_DEEP_LEARNING_NETWORK_H
