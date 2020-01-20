#!/bin/bash
#
# Resets the Docker image. Run this when you need to start with a fresh Docker image (e.g., reset any installations
# you did inside the image).

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

DOCKER_CONTAINER_NAME="spider_$USER"

cd "$PROJ_DIR"
# Stop Docker image if it's running
if [ -n "$(docker ps --filter name=$DOCKER_CONTAINER_NAME -q)" ]; then
  docker container stop "$DOCKER_CONTAINER_NAME"
fi
# Delete Docker image if it exists
if [ -n "$(docker ps -a | grep $DOCKER_CONTAINER_NAME)" ]; then
  docker container rm "$DOCKER_CONTAINER_NAME"
fi

docker run -d \
  -it \
  --name "$DOCKER_CONTAINER_NAME" \
  --mount type=bind,source="$(pwd)",target=/spider \
  --mount type=bind,source=/z/mid/D3M/datasets_public,target=/datasets \
  --mount type=bind,source=/z/mid/D3M/volumes,target=/volumes \
  --env "HOST_USER=$(id -u)" \
  --env "HOST_GROUP=$(id -g)" \
  --env "GIT_COMMITTER_NAME=$(git config user.name)" \
  --env "GIT_COMMITTER_EMAIL=$(git config user.email)" \
  --env "GIT_AUTHOR_NAME=$(git config user.name)" \
  --env "GIT_AUTHOR_EMAIL=$(git config user.email)" \
  spider
