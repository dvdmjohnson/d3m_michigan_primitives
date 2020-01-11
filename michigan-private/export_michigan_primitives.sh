#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

# Install packages required to generate D3M JSON schema
cd /
$PROJ_DIR/michigan-private/install_dependencies.sh
# Generate files to export to rszeto/primitives 
git clone https://github.com/dvdmjohnson/d3m_michigan_primitives.git
cd d3m_michigan_primitives
HASH=`git rev-parse HEAD`
SHORT_HASH=`git rev-parse --short HEAD`
python3 genjson.py

# Clone rszeto/primitives and go into newest primitives collection
cd /
git clone https://gitlab.com/rszeto/primitives.git
cd primitives/v20*
NEW_BRANCH_NAME="merge-michigan-$SHORT_HASH"
git checkout -b "$NEW_BRANCH_NAME"

# Overwrite contents of Michigan folder
if [ -d Michigan ]; then
    rm -rf Michigan
fi
mv /d3m_michigan_primitives/Michigan .

# Commit and push changes
git add Michigan
git commit -m "Update primitives from dvdmjohnson/d3m_michigan_primitives@$HASH"
git push -u origin "$NEW_BRANCH_NAME"

printf "\nCreated new branch $NEW_BRANCH_NAME at rszeto/primitives. Please create a merge request to merge this branch into datadrivendiscovery/primitives after the continuous integration tests pass.\n"
