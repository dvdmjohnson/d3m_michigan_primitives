#!/bin/bash
#
# Builds the Docker image. Run this whenever the image has been updated (e.g., with new dependencies).

SCRIPT_DIR="$(cd $(dirname $0); pwd)"

cd "$SCRIPT_DIR"
# docker login registry.datadrivendiscovery.org
docker build --pull -t "spider" -f Dockerfile .
