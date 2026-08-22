# Expression grammar definition

[Expr.g4](Expr.g4) defines the grammar for mathematical expressions
use for defining constraints, objective, etc.

[ANTLR](https://www.antlr.org) needs to be used to generate the associated
parser code, which must be written to [gems_craft.expression.parsing.antlr](/src/gems_craft/expression/parsing/antlr)
package. **No other files are expected to be present in that package**.

To achieve this you may use the provided `generate-parser.sh` script after having installed
antlr4-tools (`uv sync --group dev` in root directory).

You may also, for example, use the ANTLR4 PyCharm plugin.

We use the visitor and not the listener in order to translate ANTLR AST
into our own AST, so the options `-visitor` and `-no-listener` need to
to be used.
