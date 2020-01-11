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
  --mount type=bind,source=/z/mid/D3M/datasets,target=/datasets \
  --mount type=bind,source=/z/mid/D3M/volumes,target=/volumes \
  spider

docker exec -it spider_test bash
