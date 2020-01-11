#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$SCRIPT_DIR/.."

cd "$PROJ_DIR"
for f in `find Michigan -name *yaml`; do
    echo "Compressing $f..."
    gzip $f
done
echo "Done."

