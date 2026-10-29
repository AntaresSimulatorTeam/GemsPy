# GEMS, a graph-oriented algebraic modelling language

The [GEMS](https://gems-energy.readthedocs.io/en/latest/) framework consists of an **algebraic modelling language**, close to mathematical syntax, and a **data structure** for describing energy systems.

For extensive information regarding the language, please consult the [GEMS documentation website](https://gems-energy.readthedocs.io/en/latest/):

&rarr; [Overview](https://gems-energy.readthedocs.io/en/latest/overview/glossary/)

&rarr; [Getting started](https://gems-energy.readthedocs.io/en/latest/getting-started/installation/modeler-installation/)

&rarr; [User Guide](https://gems-energy.readthedocs.io/en/latest/user-guide/introduction/)

&rarr; [Examples](https://gems-energy.readthedocs.io/en/latest/examples/adequacy-example/)


## Power operator `^`

The binary operator `^` raises its left operand to the power of its right
operand. It follows the standard mathematical convention:

- it binds tighter than `*`, `/` and unary minus: `2*3^2` is `2*(3^2)` and
  `-2^2` is `-(2^2) = -4`;
- it is right-associative: `2^3^2` is `2^(3^2)`;
- the exponent may be signed: `2^-3` is `2^(-3)`.

The same holds inside a time shift: `x[t-2^2]` is `x[t-4]` and `x[t-2^2*3]`
is `x[t-12]`. There the exponent cannot be signed (`x[t+2^-1]` is a parse
error). `**` is not accepted.

In constraints, bounds and objective contributions, the expression must stay
linear, as for `floor`, `ceil`, `abs`, `round`, `min` and `max`: the operands
of `^` can only be parameters and literals (`p^2`, `2^p`, `p^(1 + q)`), and
`x^2` on a variable `x` is rejected with
`Non-linear expression is not allowed in ...`. In extra outputs, which are
evaluated after the solve, variables are allowed (`x^(2 + p)`).
