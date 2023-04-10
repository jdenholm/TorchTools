#!/usr/bin/bash

cd docs
make clean
make html

target_branch="doc-branch"

git ls-remote --exit-code --heads origin $target_branch
# If the target documentation branch exists, delete it
if [ $? -eq 0 ]; then git push --delete origin $target_branch; fi


# Create the documentation branch
git checkout --orphan $target_branch
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