#!/bin/bash

script_file=$(readlink -f -- "$0")
script_dir=$(dirname -- "${script_file}")
cd "${script_dir}" && antlr4 -v 4.13.2 -Dlanguage=Python3 -Werror -no-listener -visitor -o ${script_dir}/../src/gems_craft/expression/parsing/antlr Expr.g4
