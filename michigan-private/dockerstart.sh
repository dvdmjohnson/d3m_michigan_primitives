#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
docker container stop spider_test
docker container rm spider_test
docker run -d \
  -it \
  --name spider_test \
  --mount type=bind,source="$(pwd)",target=/spider \
  -e "GIT_COMMITTER_NAME=$(git config user.name)" \
  -e "GIT_COMMITTER_EMAIL=$(git config user.email)" \
  -e "GIT_AUTHOR_NAME=$(git config user.name)" \
  -e "GIT_AUTHOR_EMAIL=$(git config user.email)" \
  spider

docker exec -it spider_test bash
