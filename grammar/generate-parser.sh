#!/bin/bash

script_file=$(readlink -f -- "$0")
script_dir=$(dirname -- "${script_file}")
antlr4 -Dlanguage=Python3 -Werror -no-listener -visitor -o ${script_dir}/../src/gems/expression/parsing/antlr Expr.g4
