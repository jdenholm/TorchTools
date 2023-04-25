#!/usr/bin/bash
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"

git checkout -b gh-pages

# Build the html
cd docs
make clean
make html
cp -r _build/html/*.html .
cd ..



git add --all
git commit -m "Updated docs"
git push -u origin gh-pages --force