#!/bin/bash

PRIMITIVES_REPO_PATH="$1"

if [ "$#" != 1 ]; then
  echo "Usage: ./dockerstart_extended.sh <path to primitives repo>"
  exit 1 
fi

RESOLVED_PRIMITIVES_REPO_PATH=`realpath $PRIMITIVES_REPO_PATH`

if [ ! -d "$RESOLVED_PRIMITIVES_REPO_PATH" ]; then
  echo "$RESOLVED_PRIMITIVES_REPO_PATH (resolved from '$PRIMITIVES_REPO_PATH') is not a valid path"
  exit 2
fi

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
  --mount type=bind,source="$RESOLVED_PRIMITIVES_REPO_PATH",target=/primitives \
  spider

docker exec -it spider_test bash
