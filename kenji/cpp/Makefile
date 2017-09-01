UNAME := $(shell uname)

build:
	-mkdir build
	cd build &&\
		cmake -DBUILD_SHARED_LIBS=ON -DENABLE_INTEGRATION_TEST=ON -DBUILD_TESTS=ON -DBUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release .. &&\
		make -j 8
.PHONY: build

run:
	./build/src/cpp-deeplearning
.PHONY: run

test:
	cd build && ctest -V -j 2
.PHONY: test

install:
ifeq ($(UNAME), Linux)
	cd build && sudo make install
endif
ifeq ($(UNAME), Darwin)
	cd build && make install
endif
.PHONY: install

clean:
ifeq ($(UNAME), Linux)
	-cd build && sudo make uninstall
endif
ifeq ($(UNAME), Darwin)
	-cd build && make uninstall
endif
	-rm -rf build
.PHONY: clean
