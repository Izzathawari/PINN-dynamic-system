#!/bin/bash
set -e

branch=$(git branch --show-current)

read -r -p "Enter commit message: " commitMessage

# Exit if commitMessage is empty or contains only whitespace
if [[ -z "${commitMessage// }" ]]; then
    echo "Error: Commit message cannot be empty. Aborting."
    exit 1
fi

git add .
git commit -m "$commitMessage"
git push origin "$branch"