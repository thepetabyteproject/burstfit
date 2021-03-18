#!/bin/bash

printf "\033[0;32mGenerating markdown for docs...\033[0m\n"
pydoc-markdown
printf "\033[0;32mBuilding documentation site...\033[0m\n"
cd docs
mkdocs build