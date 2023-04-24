#!/usr/bin/bash


git config user.name "jdenholm"
git config user.email "j.denholm.2017@gmail.com"

# Build the html
cd docs
make clean
make html
cp -r _build/html/*.html .
cd ..


git add --all

git commit -m "Updated docs"
git push
