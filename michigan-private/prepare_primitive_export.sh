#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$SCRIPT_DIR/.."

# Compress pipeline run logs
cd "$PROJ_DIR"
for f in `find Michigan -name *yaml`; do
    echo "Compressing $f..."
    gzip $f
done

# Change ownership of `Michigan` to the host user
chown -R $HOST_USER:$HOST_GROUP Michigan

echo "Done."

