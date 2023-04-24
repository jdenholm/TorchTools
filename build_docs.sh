#!/usr/bin/bash


git config user.name "jdenholm"
git config user.email "j.denholm.2017@gmail.com"

current_branch="$(git branch --show-current)"
target_branch="doc-branch"

# Build the html
cd docs
make clean
make html
cp -r _build/html/*.html .
cd ..


git add --all

git commit -m "Updated docs"
git push -u origin $target_branch --force
