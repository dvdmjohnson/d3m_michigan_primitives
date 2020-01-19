#!/bin/bash
#
# Launches a bash terminal inside the Docker image. Make sure that the Docker image is running (with docker_restart.sh)
# before calling this script.

if [ -z "$(docker ps --filter 'name=spider_test' -q)" ]; then
  echo "Docker image 'spider_test' is not running. Try running './docker_reset.sh' first."
  exit 1
fi

docker exec -it spider_test bash
