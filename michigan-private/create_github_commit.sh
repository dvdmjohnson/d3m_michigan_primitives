#!/bin/bash

# List of paths to remove. Must be relative to SPIDER project root
RM_PATHS=(
)

# Get project path
SCRIPT_DIR="$(cd "$(dirname $0)"; pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.."; pwd)"

# Require arguments
if [ "$#" != 1 ]; then
    echo "Usage: create_github_commit.sh GITHUB_REPO_DIR"
    exit 1
fi

# Check that the argument is a valid path
GITHUB_REPO_DIR="$1"
if [ ! -d "$GITHUB_REPO_DIR" ]; then
    echo "$GITHUB_REPO_DIR is not a valid directory"
    exit 2
fi

# Check that the argument is a local copy of the desired repository
cd "$GITHUB_REPO_DIR"
REPO_TEST_HTTPS="$(git remote -v 2>/dev/null | grep 'https://github.com/dvdmjohnson/d3m_michigan_primitives.git')"
REPO_TEST_SSH="$(git remote -v 2>/dev/null | grep 'git@github.com:dvdmjohnson/d3m_michigan_primitives.git')"
if [[ -z "$REPO_TEST_HTTPS" && -z "$REPO_TEST_SSH" ]]; then
    echo "The argument must be a clone of dvdmjohnson/d3m_michigan_primitives on GitHub"
    exit 3
fi

# Only run if the local GitHub repo copy is clean
if [ -n "$(git diff)" ]; then
    echo "Changes were detected in the GitHub repo. Please remove them before continuing."
    exit 4
fi

# Only run if the working tree is clean
cd "$PROJ_DIR"
if [ -n "$(git diff)" ]; then
	echo "Dirty working tree detected. Please restore changed files and delete untracked files before running this script."
	exit 4
fi

# Print files that will be removed
GIT_CLEAN_OP="git clean -ffxd --dry-run"
should_proceed=false
if [ -n "$($GIT_CLEAN_OP)" ]; then
	printf "The following paths will be deleted:\n\n"
	$GIT_CLEAN_OP
    printf "\nThe above paths will be deleted and cannot be recovered. Is this okay? (y/[n])\n"
    read ans
    if [[ "${ans:0:1}" == "y" || "${ans:0:1}" == "Y" ]]; then
        should_proceed=true
    fi
else
    should_proceed=true
fi

# Exit normally if user backs out
if [ "$should_proceed" == "false" ]; then
    exit 0
fi

# Get current branch and commit hash
CUR_PROJ_BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
HASH=`git rev-parse HEAD`
SHORT_HASH=`git rev-parse --short HEAD`
# Name new branch after commit hash
NEW_BRANCH_NAME="merge-$SHORT_HASH-into-github"

# Create the new branch and push to remote
git checkout -b "$NEW_BRANCH_NAME"
for path in "${RM_PATHS[@]}"; do
    git rm -rf "$PROJ_DIR/$path"
done
git commit -m "Remove secret paths from $HASH"
git push -u origin "$NEW_BRANCH_NAME"

# Create new patched commit in GitHub repo in new branch
cd "$GITHUB_REPO_DIR"
CUR_GITHUB_BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
git checkout -b "$NEW_BRANCH_NAME"
diff -ru . "$PROJ_DIR" -x .git > patch
patch -p0 < patch
rm patch
git add -u
git commit -m "Patched $HASH into GitHub"
git push -u origin "$NEW_BRANCH_NAME"

# Return both repos to original branches
git checkout "$CUR_GITHUB_BRANCH_NAME"
cd "$PROJ_DIR"
git checkout "$CUR_PROJ_BRANCH_NAME"
