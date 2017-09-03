//
// Created by Kenji Nomura on 9/3/17.
//

#ifndef CPP_DEEPLEARNING_IMNIST_H
#define CPP_DEEPLEARNING_IMNIST_H

#include <unistd.h>

class IMnist {

 protected:
  std::int32_t magic_number;
  std::int32_t number_of_data;

  std::string getcwd_s();
  int byte4ToInt(const char buffer[], const int offset = 0);

 public:
  virtual int getNumberOfData() const = 0;
  virtual void show(const int number) const = 0;

};

inline std::string IMnist::getcwd_s() {
  char buffer[FILENAME_MAX];
  char *cwd = getcwd(buffer, FILENAME_MAX);
  if (cwd) return std::string(cwd);
  return "";
}

inline int IMnist::byte4ToInt(const char *buffer, const int offset) {
  return int((unsigned char) (buffer[offset]) << 24 |
      (unsigned char) (buffer[offset + 1]) << 16 |
      (unsigned char) (buffer[offset + 2]) << 8 |
      (unsigned char) (buffer[offset + 3]));
}

#endif //CPP_DEEPLEARNING_IMNIST_H
