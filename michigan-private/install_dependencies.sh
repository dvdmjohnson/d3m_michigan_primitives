#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
pip3 install --upgrade --upgrade-strategy only-if-needed -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@b06bc80d930e97b1c69eab860cbad35b2de17dfe#egg=common_primitives
pip3 install --upgrade --upgrade-strategy only-if-needed -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@4a2cfd1dc749bb13ce807b2bf2436a45cd49c695#egg=sklearn-wrap
pip3 install --upgrade --upgrade-strategy only-if-needed --no-cache-dir -e .

# Install nose to run individual tests
pip3 install nose
