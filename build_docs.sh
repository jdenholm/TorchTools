#!/usr/bin/bash

cd docs
make clean
make html

git checkout --orphan doc-branch
find . -type f -not -name 'docs' -delete
cd docs
cp _build/html/*.html .

cd ..
git add docs
git commit -m "Updated docs"
git push -u origin doc-branch