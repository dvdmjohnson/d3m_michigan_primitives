#!/bin/bash
#
# Resets the Docker image. Run this when you need to start with a fresh Docker image (e.g., reset any installations
# you did inside the image).

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
# Stop Docker image if it's running
if [ -n "$(docker ps --filter 'name=spider_test' -q)" ]; then
  docker container stop spider_test
fi
# Delete Docker image if it exists
if [ -n "$(docker ps -a | grep spider_test)" ]; then
  docker container rm spider_test
fi

docker run -d \
  -it \
  --name spider_test \
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
