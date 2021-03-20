#!/bin/bash

printf "\033[0;32mGenerating markdown for docs...\033[0m\n"
pydoc-markdown || exit
printf "\033[0;32mCopying example notebooks...\033[0m\n"
cp -rp examples/* docs/build/content/ || exit
cd docs || exit
printf "\033[0;32mBuilding documentation site...\033[0m\n"
mkdocs build