NAME=zero-deeplearning-van
.PHONY: build run clean

build:
	docker build -t $(NAME) .

run:
	docker run -it --rm \
		-v $(realpath .)/src:/srv/src \
		--name $(NAME) \
		$(NAME) /bin/bash

clean:
	docker ps -a -q | xargs docker rm
	docker images -f dangling=true -q | xargs -t docker rmi

