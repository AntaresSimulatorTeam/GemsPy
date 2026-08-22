# Expression syntax

Constraints, objective contributions, variable bounds, port-field definitions
and extra-outputs are all written in the GEMS expression language. This page
documents the syntax **as the GemsPy interpreter accepts it**. For the language
itself, see the
[GEMS documentation](https://gems-energy.readthedocs.io/en/latest/user-guide/introduction/).

---

## Operators

| Operator | Meaning | Notes |
|---|---|---|
| `+` `-` | addition, subtraction | `-x` is also valid as unary minus |
| `*` `/` | multiplication, division | |
| `^` | power | right-associative, binds tighter than unary minus |
| `=` `<=` `>=` | comparison | only at the top level of a constraint |

## Functions

| Function | Meaning |
|---|---|
| `floor(e)` / `ceil(e)` | round down / up to the nearest integer |
| `round(e)` | round to the nearest integer (banker's rounding) |
| `abs(e)` | absolute value |
| `min(a, b, ...)` / `max(a, b, ...)` | element-wise minimum / maximum |
| `expec(e)` | expectation over scenarios |
| `sum(e)` | sum over all timesteps |
| `sum(t-a..t+b, e)` | sum over a time window |
| `sum_connections(port.field)` | sum of a port field over all connections |
| `dual(constraint_id)` | dual value of a constraint (extra-outputs only) |
| `reduced_cost(variable_id)` | reduced cost of a variable (extra-outputs only) |

## Time indexing

| Syntax | Meaning |
|---|---|
| `x[t]` | value at the current timestep (equivalent to `x`) |
| `x[t-1]`, `x[t+d]` | value at a timestep relative to the current one |
| `x[3]` | value at an absolute timestep |
| `(expr)[t-1]` | shift applied to a whole expression |

---

## Precedence

From tightest to loosest binding:

1. `^`
2. unary `-`
3. `*` `/`
4. `+` `-`
5. `=` `<=` `>=`

`^` is **right-associative**; all the other binary operators are
left-associative. Parentheses override precedence anywhere.

### The power operator and unary minus

`^` binds **tighter than unary minus**, following standard mathematical
notation:

| Expression | Parses as | Value |
|---|---|---|
| `-2^2` | `-(2^2)` | `-4` |
| `2^-3` | `2^(-3)` | `0.125` |
| `2^3^2` | `2^(3^2)` | `512` |
| `2*3^2` | `2*(3^2)` | `18` |
| `2^3*2` | `(2^3)*2` | `16` |

!!! warning "Divergence from Antares Simulator"
    Antares Simulator currently parses `-2^2` as `(-2)^2 = 4`. GemsPy follows
    the standard convention and evaluates it to `-4`. Write `(-2)^2`
    explicitly if you need the same result in both interpreters.

### Inside time-shift brackets

Within `x[t ...]`, the sign applies to the whole power operand and `^` does not
swallow a trailing `*` or `/`:

| Expression | Shift amount |
|---|---|
| `x[t-2^2]` | `-4` |
| `x[t-2^2*3]` | `-12` |
| `x[t-2*3^2]` | `-18` |

A **signed exponent is not allowed** inside a time shift: `x[t+2^-1]` is a
parse error, since a fractional shift is meaningless. `2^-1` remains valid
everywhere else.

---

## Where nonlinear expressions are allowed

`^`, `floor`, `ceil`, `abs`, `round`, `min` and `max` may be applied to
**literals and parameters** anywhere. Applying them to a **decision variable**
produces a nonlinear expression, which is:

- **rejected** in constraints, objective contributions and variable bounds,
  with `Non-linear expression is not allowed in ...`;
- **allowed** in `extra-outputs`, which are evaluated after the solve.

So `p^2`, `2^p` and `p^(1+q)` are valid in a constraint, `x^2` is not, and
`x^(2 + p)` is valid in an extra-output.

The **exponent must never depend on a decision variable**, in any context:
`p^x` raises `Exponent of a power expression must not depend on variables.`

~~~ yaml
constraints:
  - id: capacity
    expression: gen <= p_max^2      # parameters only — fine

extra-outputs:
  - id: squared_generation
    expression: gen^2               # variable base — post-solve only
~~~

!!! note
    `**` is not an alias for `^`. GEMS accepts `^` only, matching Antares
    Simulator. (The GemsPy *Python* API does support `param("p") ** 2`, since
    that is Python's own operator.)
