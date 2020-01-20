#!/bin/bash
#
# Builds the Docker image. Run this whenever the image has been updated (e.g., with new dependencies).

SCRIPT_DIR="$(cd $(dirname $0); pwd)"

DOCKER_IMAGE_NAME="spider_${USER}_image"

cd "$SCRIPT_DIR"
# docker login registry.datadrivendiscovery.org
docker build --pull -t "$DOCKER_IMAGE_NAME" -f Dockerfile .
