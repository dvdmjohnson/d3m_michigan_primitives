#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
pip3 install -e git+https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git@697dabc03c46c1900483bea89d576e82b5a5e4c5#egg=bbn_primitives
pip3 install --upgrade-strategy only-if-needed -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@352233f9d8e1bf813e7f0150ac9f6480df16538f#egg=common_primitives
pip3 install --upgrade --upgrade-strategy only-if-needed --no-cache-dir -e .
pip3 install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@889f93af9439fbeb29db961b03e46eaa9e2a7888#egg=sklearn-wrap

