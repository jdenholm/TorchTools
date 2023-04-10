#!/usr/bin/bash

cd docs
make clean
make html

target_branch="doc-branch"


# Create the documentation branch
git checkout $target_branch
shopt -s extglob
git rm -rf !("docs")


cd docs
cp _build/html/*.html .

cd ..
git add --all

git config user.name "jdenholm"
git config user.email "j.denholm.2017@gmail.com"

git commit -m "Updated docs"
git push -u origin $target_branch