#!/usr/bin/bash


git config user.name "jdenholm"
git config user.email "j.denholm.2017@gmail.com"

source_branch=$GITHUB_HEAD_REF
target_branch=$GITHUB_BASE_REF


# Build the html
cd docs
make clean
make html
cp -r _build/html/*.html .
cd ..


# Note: if you are merging from dev to main, source_branch would be dev and
# target branch would be main.



git add --all

git commit -m "Updated docs"
git push -u origin gh-pages