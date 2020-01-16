#!/bin/bash

if [ "$#" != 1 ]; then
  echo "Usage: ./export_michigan_primitives.sh <path to local copy of rszeto/primitives"
  exit 1
fi

# Check for and resolve given primitives path
PRIMITIVES_REPO_PATH="$1"
if [ ! -d "$PRIMITIVES_REPO_PATH" ]; then
  echo "Could not find path $PRIMITIVES_REPO_PATH"
  exit 1
fi
PRIMITIVES_REPO_PATH="$(cd $PRIMITIVES_REPO_PATH; pwd)"

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

# Check that the `Michigan` folder was generated
cd "$PROJ_DIR"
if [ ! -d "$PROJ_DIR" ]; then
  echo "Could not find `Michigan/` under $PROJ_DIR"
  exit 1
fi

# Get this repository's hash
cd "$PROJ_DIR"
HASH=`git rev-parse HEAD`
SHORT_HASH=`git rev-parse --short HEAD`

# Update primitives repo
cd "$PRIMITIVES_REPO_PATH"
git checkout master
git pull
git fetch d3m && git merge d3m/master
git push

# Go into newest primitives collection
cd $PRIMITIVES_REPO_PATH/v20*
# Overwrite contents of Michigan folder
if [ -d Michigan ]; then
    rm -rf Michigan
fi
cp -r "$PROJ_DIR/Michigan" .

# Commit and push changes
cd $PRIMITIVES_REPO_PATH
NEW_BRANCH_NAME="merge-michigan-$SHORT_HASH"
git checkout -b "$NEW_BRANCH_NAME"
# Adds files larger than 100K with lfs
./git-add.sh  
git add Michigan
git commit -m "Update primitives from dvdmjohnson/d3m_michigan_primitives@$HASH"
git push -u origin "$NEW_BRANCH_NAME"
git checkout master
git branch -d "$NEW_BRANCH_NAME"

printf "\nCreated new branch $NEW_BRANCH_NAME at rszeto/primitives. Please create a merge request to merge this branch into datadrivendiscovery/primitives after the continuous integration tests pass. In addition, remove the 'Michigan' folder from d3m_michigan_primitives if you are done with it.\n"

