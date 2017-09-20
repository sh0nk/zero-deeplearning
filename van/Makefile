NAME=zero-deeplearning-van
.PHONY: build run test clean

build:
	docker build -t $(NAME) .

run:
	docker run -it --rm \
		-v $(realpath .)/src:/srv/src \
		--name $(NAME) \
		$(NAME) /bin/bash

attach:
	docker exec -it $(NAME) /bin/bash

test:
	docker run -it --rm \
		-v $(realpath .)/src:/srv/src \
		--name $(NAME) \
		$(NAME) /srv/src/run-test.sh

clean:
	docker ps -a -q | xargs docker rm
	docker images -f dangling=true -q | xargs -t docker rmi

