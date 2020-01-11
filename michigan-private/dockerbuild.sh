#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"

cd "$SCRIPT_DIR"
# docker login registry.datadrivendiscovery.org
docker build --pull -t "spider" -f Dockerfile .
