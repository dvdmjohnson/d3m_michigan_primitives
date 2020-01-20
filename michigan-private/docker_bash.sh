#!/bin/bash
#
# Launches a bash terminal inside the Docker image. Make sure that the Docker image is running (with docker_restart.sh)
# before calling this script.

DOCKER_CONTAINER_NAME="spider_$USER"

if [ -z "$(docker ps --filter name=$DOCKER_CONTAINER_NAME -q)" ]; then
  echo "Docker image $DOCKER_CONTAINER_NAME is not running. Try running './docker_reset.sh' first."
  exit 1
fi

docker exec -it "$DOCKER_CONTAINER_NAME" bash
