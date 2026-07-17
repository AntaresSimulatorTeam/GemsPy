# Custom Sets and Indexing — GemsPy Implementation Strategy

## Status and scope

This document turns the **design proposal** already merged into the GEMS documentation repo
(`GEMS/doc/user-guide/mathematical-syntax.md` §"Custom Sets and Indexing (Proposed)", plus the
corresponding sections in `library.md`, `system.md`, `data-series.md`, `outputs/simulation-table.md`)
into a concrete GemsPy code implementation plan. It is organized into **5 PRs**, meant to land
**strictly in the order listed** — later PRs depend on earlier ones, even though they were planned in
parallel.

It was produced by three parallel planning passes (grammar/AST, model/library/system schema +
semantic indexing, data I/O + solver vectorization), then reconciled by hand. The **"Cross-validation
findings"** section below is that reconciliation — read it before the per-PR sections, since it
resolves naming mismatches and a couple of assumptions the three passes made independently.

No code has been written yet. This is a plan for a human engineer (or a future agent session) to
execute PR by PR.

---

## PR organization at a glance

| # | PR | Depends on | Risk | Can it be deferred? |
|---|----|----|----|----|
| 1 | Grammar & AST (`{ }` syntax, `sum_over` parsing) | — | Low | No — foundational |
| 2 | Model/Library/System schema (`sets`, `indexed-by` declarations) | — (parallel-safe with #1, see below) | Low | No — foundational |
| 3 | Semantic indexing resolution (`IndexingStructure` extension, AST rewrite pass) | #1, #2 | **Medium-high** — widest blast radius | No — nothing solves without it |
| 4 | Data I/O + solver vectorization for **global** (uniform) sets | #3 | Medium | No — this is the MVP payoff |
| 5 | Solver support for **local** (ragged) sets | #4 | **High** — genuinely new array-shape problem | **Yes** — ship #1-4 first, local sets as a fast-follow |

PRs #1 and #2 touch almost disjoint files and have no logical dependency on each other, so they can be
developed and reviewed concurrently — but PR #2's `resolve_library.py` changes populate a field
(`ModelIdentifiers.sets`) that PR #1 introduces, so PR #1 should **merge first** even if both are
ready at the same time (a one-line rebase, not a redesign).

---

## Cross-validation findings

Three points needed reconciling across the three planning passes; all are already reflected in the
per-PR sections below, but are called out here explicitly since this is exactly what parallel
planning risks getting wrong.

### 1. Naming mismatch: `SumOverNode`, not `SetSumNode`

The AST/grammar plan and the schema/semantics plan both independently settled on **`SumOverNode(UnaryOperatorNode)`** with a `set_id: str` field, for `sum_over(set_id, expr)`. The solver-vectorization plan was written without visibility into that decision and used a placeholder name, `SetSumNode`. **Resolution: the class is `SumOverNode`.** PR #4's `VectorizedBuilderBase` method should be named `sum_over(self, node: SumOverNode)` — matching the ABC's existing convention of one lowercase-snake method per node kind (`time_eval`, `all_time_sum`, `scenario_operator`) — not `set_sum`.

### 2. The solver-vectorization plan's three open questions are all answered by the schema/semantics plan

The solver plan flagged three things it couldn't fully pin down without seeing the semantics layer's
output. All three are now answered:

- *"I don't know the exact node names/shapes for current-element / explicit-index / shift / named-element."* → They are `SetPositionNode(set_id: str)` (bare set-id-as-value), `SetShiftNode(operand, set_id, shift)` (mirrors `TimeShiftNode`), and `SetExplicitIndexNode(operand, set_id, position)` (mirrors `TimeEvalNode`). PR #4 should implement `VectorizedBuilderBase.set_position`/`set_shift`/`set_explicit_index` as the direct structural analogues of however `time_shift`/`time_eval` are implemented today — same constant-fast-path / symbolic-parameter-slow-path split, not a new pattern.
- *"If the other agent instead resolves indices to a single already-baked `.isel`/`.sel` call before handing me the AST, my visitor would be simpler."* → It does not: `SetExplicitIndexNode.position` is kept **symbolic** when it's a parameter reference (e.g. `X[parameter_idx]`-style), evaluated at vectorization time — exactly like `TimeEvalNode` already works. Only named-element access (`X{gas}`) gets pre-baked to a `LiteralNode` integer position at library-load time, since the element list is already known then.
- *Attachment point for `set_domains`.* → It's `Component.resolved_sets: Dict[str, SetDomain]` (not a bare free-floating `set_domains` structure) — confirmed exactly as assumed on the `SetDomain`/`OrdinalSetDomain`/`EnumeratedSetDomain` shapes, just with a concrete home.

### 3. `IndexingStructure.sets` ordering: use it for membership only, not for column/dim order

The schema/semantics plan made `IndexingStructure.sets` equality and hashing **order-independent**
(`frozenset`-based), even though the field is stored as an ordered tuple — because the same logical
structure can accumulate its `sets` tuple in different orders depending on sub-expression traversal
order (`a + b` vs `b + a`). **Consequence for PR #4:** don't use `IndexingStructure.sets`'s order to
decide array-dimension order, tidy-CSV column order, or `set_id`/`set_index` comma-join order — use
`Parameter.indexed_by` / `Variable.indexed_by` / `Constraint.indexed_by` instead, which preserve true
declaration order exactly. This one-line correction should be applied to PR #4's implementation before
any dim-ordering code is written, since the original solver plan assumed `IndexingStructure.sets`
itself was the ordering source.

### 4. A breaking change neither downstream plan knew about yet — now fully contained

The schema/semantics plan introduces a wrapper, `IndexedExpression(expression, indexed_by)`, replacing
the bare `ExpressionNode` currently stored as the dict value in `Model.objective_contributions` and
`Model.extra_outputs` (needed so an objective-contribution or extra-output can carry its own
`indexed-by` field). This breaks exactly two call sites that live in PR #4's territory:
`src/gems/simulation/optimization.py:749` and `src/gems/simulation/simulation_table.py:303`, both of
which currently do `for id, expr in model.objective_contributions.items()` / `...extra_outputs...` and
use `expr` directly — both need `.expression` added. **Both exact line numbers were independently
identified by the solver-vectorization plan for unrelated reasons before this dependency was known**,
which is a reassuring cross-check that no other call site was missed. PR #4's description should
explicitly call out this two-line fix as part of its scope (not a surprise mid-review).

### 5. No circularity between local-set cardinality-by-parameter and the data loader

The solver plan flagged a risk: if a local ordinal set's `cardinality` references a scalar parameter,
and `Component.resolved_sets` needs that parameter's concrete value, does resolving `Component.resolved_sets`
require the data-series loader (`DataBase`) to already be built — creating a circular dependency? It
does not: a `cardinality`-referencing parameter is required to be scalar (`time-dependent: false`,
`scenario-dependent: false`), which means its value is given **directly inline** in `system.yml`'s
`ComponentSchema.parameters[...].value` field (a plain number), not via a data-series CSV. The existing
call order already resolves components (and can therefore resolve `Component.resolved_sets`) **before**
`build_data_base` runs, exactly like it already resolves every other component-level scalar parameter
value before data loading. No new sequencing work is needed in PR #2/#3; this is confirmed safe by the
existing architecture, not something to design around.

---

## PR #1 — Grammar & AST layer

**Goal:** GEMS expressions using `{ }` and `sum_over` parse into a well-formed AST. No model/library
schema changes, no semantic meaning attached yet — purely syntax.

**Grammar changes (`grammar/Expr.g4`):** four new `expr` alternatives, reusing the existing `argList`
rule verbatim (so both `X{segment}` and `X{segment, fuel}` fall out for free, no new comma-handling):

```antlr
| IDENTIFIER '{' argList '}'                    # setIndex
| '(' expr ')' '{' argList '}'                  # setIndexExpr
| IDENTIFIER '{' argList '}' '[' shift ']'       # setIndexTimeShift
| IDENTIFIER '{' argList '}' '[' expr ']'        # setIndexTimeIndex
```

No lexer changes (`{`/`}` become implicit tokens exactly like `(`/`)`/`,` already are). No ambiguity
with `[ ]` (disjoint delimiters) or with the generic `function` rule (disambiguated by the very next
token, `(` vs `{`, the same way ANTLR already disambiguates `X(...)` from `X[...]`). `sum_over` needs
**zero grammar changes** — it reuses the existing generic `IDENTIFIER '(' argList? ')'  # function`
production, exactly like `expec`/`dual`/`reduced_cost` already do; it's recognized purely by name
inside `visitFunction`, following the exact existing pattern for `dual`/`reduced_cost` (whose first
argument is a raw name, not a value to evaluate).

Composing set-indexing with time-indexing without parens (`X{segment}[t+1]`, matching the spec's own
example table) needs the two extra `setIndexTimeShift`/`setIndexTimeIndex` alternatives above; any
other chaining (`(X{segment}){fuel}`, indexing an already-indexed expression by a second set) requires
parens, exactly like chained time-brackets already require parens today (`(X[t-1])[t-1]`, not
`X[t-1][t-1]`).

**New AST nodes (`src/gems/expression/expression.py`)** — one generic, semantics-free carrier per
concept, with all disambiguation (current-element vs. explicit index vs. shift vs. named element)
deferred to PR #3, since the grammar has no visibility into which identifiers are declared set ids
(unlike `t`, which is a dedicated reserved lexer token):

```python
@dataclass(frozen=True, eq=False)
class SetIndexNode(UnaryOperatorNode):
    index_exprs: Tuple[ExpressionNode, ...]   # one raw sub-expression per `{}` slot, source order

@dataclass(frozen=True, eq=False)
class SumOverNode(UnaryOperatorNode):
    set_id: str                                # raw text of sum_over's first argument, unvalidated here

@dataclass(frozen=True, eq=False)
class IdentifierNode(ExpressionNode):
    name: str                                  # bare identifier that isn't a known variable/parameter
```

`ExpressionNode` gains `.set_index(*index_exprs)` / `.sum_over(set_id)` builder methods, mirroring
`.shift()`/`.eval()`/`.time_sum()`.

**Parser changes (`src/gems/expression/parsing/parse_expression.py`):**
- `ModelIdentifiers` gains `sets: Set[str] = field(default_factory=set)` and `is_set()` — empty by
  default, so this is purely additive until PR #2 starts populating it.
- `_convert_identifier` becomes an overridable hook: known variable/parameter → unchanged; known set
  → `IdentifierNode`; otherwise → still raises (regression guard: ordinary typos keep failing loudly).
- A separate, always-lenient identifier-conversion path is used **only** while building `index_exprs`
  (i.e. inside `{ }`), since a name there might be an enumerated element (`gas`) that the parser has no
  way to validate — it always falls back to `IdentifierNode` rather than raising.
- `visitSetIndex` / `visitSetIndexExpr` / `visitSetIndexTimeShift` / `visitSetIndexTimeIndex` added,
  mirroring `visitTimeShift`/`visitTimeIndex` line-for-line.
- `visitFunction` gains a `sum_over` special case (checked before eagerly visiting arguments, exactly
  like the existing `dual`/`reduced_cost` special cases): first argument's raw text becomes `set_id`,
  second argument is visited normally. Nested `sum_over(fuel, sum_over(segment, X))` needs no special
  handling — ordinary recursion already produces it.

**`ExpressionVisitor` blast radius:** `visitor.py` gains 3 new abstract methods
(`set_index`/`sum_over`/`identifier`), which breaks every existing concrete subclass until each gets a
stub or real implementation. Real implementations land in this PR for: `print.py`, `copy.py`,
`degree.py` (indexing/aggregation doesn't add nonlinearity — `identifier` is always degree 0, a bare
set-id is always a constant integer — these are correct final answers, not stubs), `equality.py`,
`uses_sum_connections_on.py`. Deliberate `raise NotImplementedError`/`ValueError` stubs (matching each
file's own existing precedent for not-yet-resolvable nodes like `port_field`) land for: `evaluate.py`,
`indexing.py`'s `TimeScenarioIndexingVisitor` (message: "must be resolved before computing time/scenario
indexing structure" — PR #3 replaces the `sum_over` half of this stub with real logic and leaves
`set_index`/`identifier` as **permanent** raises, since after PR #3 those two node types never appear
inside a constructed `Model` at all), `vectorized_builder.py`'s three visitor classes (PR #4 replaces
these), and two small existing model-layer visitors (`model/port.py`, `model/resolve_library.py`) that
just need trivial recurse-into-operand stubs.

**Parser regeneration:** edit `Expr.g4`, run `grammar/generate-parser.sh` (needs `antlr4-tools`,
already a pinned `dev` dependency — no new environment prerequisite), commit the regenerated
`src/gems/expression/parsing/antlr/*` output alongside the `.g4` diff in the same commit (per
`grammar/README.md`'s "no other files are expected in that package, never hand-edit" convention).

**Testing:** extend `tests/unittests/expressions/parsing/test_expression_parsing.py`'s existing
`@pytest.mark.parametrize` tables — positive cases for every new grammar form (`X{segment}`, `X{2}`,
`X{segment+1}`, `X{gas}`, `X{segment, fuel}`, `(a+b){segment}`, `X{segment}[t+1]`,
`(X{segment})[1]`, standalone bare `segment` as a value, `sum_over(...)` including nested); negative
cases (`X{}` empty braces, unparenthesized `X{segment}{fuel}` chaining, a genuine unknown-identifier
typo still raising). Add real-visitor coverage to `test_printer.py`/`test_copy.py`/`test_equality.py`/
`test_degree.py`, and a stub-consistency test in `test_evaluation.py` mirroring the existing
`test_dual_reduced_cost_evaluation_raises` pattern.

**Files touched:** `grammar/Expr.g4`; `src/gems/expression/parsing/antlr/*` (regenerated);
`src/gems/expression/expression.py`; `src/gems/expression/visitor.py`;
`src/gems/expression/parsing/parse_expression.py`; `src/gems/expression/{print,copy,degree,equality,
uses_sum_connections_on,evaluate,indexing}.py`; `src/gems/model/port.py`;
`src/gems/model/resolve_library.py`; `src/gems/simulation/vectorized_builder.py` (stubs only); test
files listed above.

---

## PR #2 — Model / Library / System schema layer

**Goal:** `sets` and `indexed-by` are declarable and validated in library files and system files. Pure
data-model + YAML parsing + reference/naming validation — **no expression-tree involvement at all**
(`indexed-by` is just a list of strings at this layer), so this PR has no dependency on PR #1 landing
first, only a light one-field interaction (see below).

**New dataclasses (`src/gems/model/set.py`, new file):**

```python
class SetScope(Enum): LOCAL = "LOCAL"; GLOBAL = "GLOBAL"
class SetKind(Enum): ORDINAL = "ORDINAL"; ENUMERATED = "ENUMERATED"

@dataclass(frozen=True)
class SetDeclaration:
    id: str
    scope: SetScope
    kind: SetKind
    description: Optional[str] = None
    cardinality_parameter: Optional[str] = None   # LOCAL + ORDINAL only; names a scalar parameter

@dataclass(frozen=True)
class OrdinalSetDomain:
    cardinality: int

@dataclass(frozen=True)
class EnumeratedSetDomain:
    elements: Tuple[str, ...]

SetDomain = Union[OrdinalSetDomain, EnumeratedSetDomain]
```

One dataclass for `SetDeclaration` (not a 2×2 local/global × ordinal/enumerated class hierarchy) — a
`scope`/`kind` pair of enums is enough, and avoids forcing every generic piece of downstream code to
special-case four types.

**A `SetDeclaration` never carries a concrete value, for either scope, full stop** — this was a design
fork resolved in favor of full uniformity: local sets used to optionally allow a literal `cardinality`
or directly-given `elements` in the model, but that option is now removed, so `SetDeclaration` has no
`cardinality`/`elements` fields at all (only `cardinality_parameter`, which is a *reference*, not a
value, and only ever set for a local ordinal set). There is therefore no `is_resolved()` method to
reason about anymore — a `SetDeclaration` built from `Model.sets`/`Library.sets` is a pure structural
declaration, permanently. The sole source of concrete values is a separate structure,
`Component.resolved_sets: Dict[str, SetDomain]` (below), built fresh by `resolve_components.py` from
`system.yml`. Anything needing a set's concrete value at solve/vectorization time (PR #4) reads
`Component.resolved_sets`, never a `SetDeclaration`.

`Model.sets: Dict[str, SetDeclaration]` (local) and `Library.sets: Dict[str, SetDeclaration]` (global,
a new top-level collection sibling to `port-types`/`models`, per the doc). `PortField` gains
`indexed_by: Tuple[str, ...] = ()`.

**Two YAML schemas, nearly identical, since local and global sets now differ by exactly one field**
(an earlier version of this plan gave local sets more flexibility than global sets — `cardinality` as
either a literal or a parameter reference, and `elements` given directly — both now removed, per the
doc's full alignment: *no* set, local or global, ever gives a concrete value in the library or model):

- `GlobalSetSchema` (used in `LibrarySchema.sets`): `id`, `description`, `kind` (`Literal["ordinal",
  "enumerated"]`, **mandatory**). No `cardinality`/`elements` field at all.
- `LocalSetSchema` (used in `ModelSchema.sets`): identical, plus one additional field —
  `cardinality` (`Optional[str]`, a scalar-parameter id — **never a literal integer**), required when
  `kind: ordinal` and forbidden when `kind: enumerated`. There is no `elements` field here at all; an
  enumerated local set's concrete elements are always supplied per component in `system.yml` (see
  below), exactly like a global set's are always supplied once, study-wide.

Both schemas use Pydantic `extra="forbid"` so a stray `cardinality`/`elements` key structurally cannot
appear where it isn't allowed (e.g. `elements` on any set declaration, or a literal `cardinality` on a
local one) — enforced by the schema shape itself, not a runtime check that would need to guess intent.

This also resolves what an earlier version of this plan flagged as a spec ambiguity (the `kind` of a
fully-deferred global set, since neither `cardinality` nor `elements` was ever given at library level)
— `kind` is now a required field on both schemas, so there's no ambiguity or default to reason about;
every set states its kind explicitly, always.

An `indexed_by` field (`Optional[Union[str, List[str]]]`, normalized to a tuple immediately after
parsing) is added to `ParameterSchema`, `VariableSchema`, `ConstraintSchema` (covers constraints and
binding-constraints — same schema class), `ObjectiveContributionSchema`, `ExtraOutputSchema`, and the
port-type `FieldSchema`.

**Validation, at library-load time (`resolve_library.py`), before any `Model`/`Library` is
constructed:**
1. Naming collisions: a local set id can't collide with a parameter/variable id in the same model or
   the literal `t`; a **global** set id likewise can't be the literal `t` either (checked when
   `Library.sets` is built, since a global set's id is just as usable bare as a local set's). More
   generally, **no locally-declared id in a model — parameter, variable, local set, port, constraint,
   binding-constraint, objective-contribution, or extra-output — may collide with any global set id
   visible in that library**, since a global set's id is resolvable bare from inside any model without
   local declaration (via `indexed-by` or a bare current-position reference), creating the same
   ambiguity a local-set/parameter collision would. Implemented as two checks: (a) library-load time,
   `Library.sets`' ids must not include `t`; (b) per-model, compute `all_visible_ids` (every
   parameter/variable/local-set/port/constraint/binding-constraint/objective-contribution/extra-output
   id declared in the model) and validate it has no overlap with the library's global set ids, in
   addition to the narrower local-set-vs-parameter/`t` check.
2. Every `indexed-by` entry resolves to a real, visible set (local ∪ global).
3. A local ordinal set's `cardinality`-parameter reference must itself be scalar and not itself
   `indexed-by` anything (no circular set-size dependency).
4. No `SetDeclaration`, local or global, ever carries a concrete `cardinality`/`elements` value —
   enforced structurally by both `GlobalSetSchema` and `LocalSetSchema` having no such fields at all
   (`extra="forbid"`), not by a runtime check. Every set's concrete value is resolved later, exclusively
   by `resolve_components.py`, from `system.yml` (see below).
5. Port-field-definition indexing consistency (a field's declared `indexed-by` must match its
   definition's inferred structure) — this specific check is deferred to PR #3, since it needs
   `compute_indexation`, but the schema plumbing that makes it checkable belongs here.

**Port-crossing enforcement:** almost entirely free — a port-type's `indexed-by` is resolved
exclusively against `Library.sets` because no `Model` (and therefore no local `SetDeclaration`) exists
yet at the point port types are parsed. One explicit check is still added: every id in a port field's
`indexed-by` must be a key in the library's already-resolved global `sets` dict, or raise clearly
rather than silently producing an empty lookup.

**`system.yml` schema (`src/gems/study/parsing.py`):** `SystemSetSchema` (`id`, `cardinality`,
`elements`) in a new top-level `sets:` list (sibling to `components`/`connections`), for **study-wide**
instantiation of **every** library-level global set (no global set is ever resolved any other way);
`ComponentSetSchema` (`id`, `elements`) in a new per-component `sets:` list, for per-component
instantiation of **every** local enumerated set any instantiated model declares (no local enumerated
set is ever resolved any other way either — mirrors how `properties` values are supplied per-component
while their keys are declared in the model).

**Resolution (`src/gems/study/resolve_components.py`):** for every library-level global set declared by
any library the system uses, look it up in `SystemSchema.sets`; error if it's missing, and error if the
entry's shape (`cardinality` vs. `elements`) doesn't match the library's declared `kind` for that set.
For every local ordinal set, its size comes from the ordinary per-component parameter-assignment
mechanism (the model's `cardinality` field just names which parameter — no separate `sets:` entry is
involved). For every local enumerated set, every instantiating component must supply matching elements
in its own `ComponentSetSchema.sets` entry; error listing missing ids, in the same style as the
existing missing-parameter/missing-property checks. `Component` gains `resolved_sets: Dict[str,
SetDomain] = field(default_factory=dict)` — local sets specific to that component, plus every global
set of its model, fully concrete. `Study` gains a `check_set_consistency()` validation, mirroring the
existing time/scenario data-requirement check.

**Interaction with PR #1:** this PR's `resolve_library.py` changes populate `ModelIdentifiers.sets`
(the field PR #1 introduces, empty by default) with the union of a model's local set ids and every
global set visible in its library, right before `parse_expression()` is called. This is the one
concrete reason PR #1 should merge first (or this PR should rebase on top of it) — otherwise no
logical dependency exists between the two.

**Testing:** `tests/unittests/lib_parsing/` — round-trip parsing of an ordinal and an enumerated global
set (`GlobalSetSchema`, `kind`-only, no `cardinality`/`elements` accepted — assert a stray
`cardinality`/`elements` key on a library-level set entry is rejected), a local ordinal set with a
parameter-referenced cardinality (plus the scalar/non-circular validation errors, and a rejection test
for a literal integer `cardinality` on a local set — `LocalSetSchema.cardinality` is string-only), a
local enumerated set (asserting it never accepts an `elements` key — `LocalSetSchema` has none —
always resolved in `system.yml`), a port-type field indexed by a global set (plus the local-id-rejected
error case), and the naming-collision error cases: a local set id equal to `t`, a local set id
colliding with a parameter/variable id, a global set id equal to `t`, and — covering the full breadth
of the broadened rule, not just one representative case — a global-set-id collision for *each* other
locally-declared entity kind in turn (a variable, a port, a constraint, a binding-constraint, an
objective-contribution, and an extra-output each separately named the same as a visible global set).
`tests/unittests/system/` — `system.yml`'s two new `sets:` blocks resolving correctly for
both an ordinal and an enumerated global set, a missing-global-set-entry error, a
kind-mismatch error (e.g. `elements` given in `system.yml` for a set the library declared
`kind: ordinal`), missing-local-set-elements error.

**Files touched:** new `src/gems/model/set.py`; modify `src/gems/model/{model,library,port,parsing,
resolve_library,__init__}.py`; `src/gems/study/{parsing,system,resolve_components,study}.py`; test
files listed above.

---

## PR #3 — Semantic indexing resolution

**Goal:** the layer where custom-set indexing actually acquires meaning: `IndexingStructure` grows a
third dimension, and the raw `SetIndexNode`/`IdentifierNode` carriers from PR #1 get rewritten, using
PR #2's set declarations, into concrete, permanent AST node types every downstream consumer can rely
on. **This is the highest-risk PR** — it touches the most files across the existing codebase and is
where an implementation mistake would be easiest to make, so budget the most review time here.

**`IndexingStructure` (`src/gems/expression/indexing_structure.py`):**

```python
@dataclass(frozen=True, eq=False)
class IndexingStructure:
    time: bool
    scenario: bool
    sets: Tuple[str, ...] = ()   # order-preserving storage; see note below

    def __eq__(self, other): ...   # frozenset(self.sets) == frozenset(other.sets) — order-independent
    def __hash__(self): ...        # consistent with __eq__
    def __or__(self, other): ...   # union of time/scenario/sets, first-seen order, deduped
```

The default empty tuple makes this additive to all ~30+ existing 2-positional-argument construction
sites across the codebase and its tests — no required edits there. Real call-site changes are needed
in `src/gems/expression/indexing.py` (the `TimeScenarioIndexingVisitor`): `variable`/`parameter` must
pass through the stored structure's `.sets` rather than reconstructing only the two booleans;
`time_eval`/`all_time_sum`/`scenario_operator` must pass `.sets` through unchanged (they only ever
collapse time or scenario respectively); `model.py`'s existing `sum_connections`-dual fallback
(`IndexingStructure(time=True, scenario=True)`) should keep `sets=()` — a known, pre-existing
conservative approximation, not a new regression.

**Reminder (from cross-validation finding #3): don't use `IndexingStructure.sets`'s order for anything
except membership testing** — canonical order lives on `Parameter.indexed_by`/`Variable.indexed_by`/
`Constraint.indexed_by` instead.

**The resolution pass (new `src/gems/model/set_resolution.py`)** runs immediately after
`parse_expression()` and before any `Constraint`/`Variable`/`IndexedExpression`/`PortFieldDefinition` is
constructed, for every expression-bearing field. It's a `CopyVisitor`-shaped rewrite (recurse and
reconstruct everything, override 3 node kinds):

- `SumOverNode` → validate `set_id` is a known set; otherwise pass through unchanged (needs no
  rewriting — see PR #1/#3 naming note above, this stays `SumOverNode` permanently).
- A standalone `IdentifierNode` (not inside a `{}` slot) → must name a known set, else error; rewritten
  to `SetPositionNode(set_id)`.
- `SetIndexNode(operand, index_exprs)` → recurse into `operand` first, then classify each index slot:
  a bare reference to the set at that declared position → drop the slot (current-element is a semantic
  no-op, exactly like `X[t] == X`); an arithmetic combination of that reference with a constant-foldable
  delta → `SetShiftNode(operand, set_id, shift)`; a bare reference to a *named element* of an already-
  resolved enumerated set → `SetExplicitIndexNode(operand, set_id, position=LiteralNode(index))`; a
  bare literal or scalar-parameter expression → `SetExplicitIndexNode(operand, set_id, position=<that
  expression>)`, kept symbolic if it's a parameter reference (evaluated later, exactly like
  `X[parameter_idx]` already works for time). Multiple remaining (non-dropped) slots nest, in written
  order. If every slot drops, the whole node degenerates to the bare, resolved `operand`.

New permanent node types (`src/gems/expression/expression.py`):

```python
@dataclass(frozen=True, eq=False)
class SetPositionNode(ExpressionNode):
    set_id: str

@dataclass(frozen=True, eq=False)
class SetShiftNode(UnaryOperatorNode):
    set_id: str
    shift: ExpressionNode

@dataclass(frozen=True, eq=False)
class SetExplicitIndexNode(UnaryOperatorNode):
    set_id: str
    position: ExpressionNode
```

`SetIndexNode` and `IdentifierNode` (from PR #1) become **transient** — they exist only between
parsing and resolution, never inside a constructed `Model`. Every visitor's stub for them (added in
PR #1) stays a **permanent** raise; only `sum_over` and the three new node types need real
implementations added to `indexing.py` in this PR, following the exact `time_sum`/`scenario_operator`
collapse-one-dimension pattern.

**Named-element access is never resolvable, for any set — this is now a permanent, structural
restriction, not a conditional one:** named-element access (`X{gas}`) can only resolve eagerly if the
target enumerated set's `elements` are already known when `resolve_set_indexing` runs. Since *no* set
— local or global, ordinal or enumerated — ever gives concrete `elements` in the library or model (see
PR #2: `LocalSetSchema`/`GlobalSetSchema` have no `elements` field at all; every enumerated set's
elements are always resolved later, from `system.yml`, either once study-wide or per component), this
case can never be resolved, for any set, ever. `resolve_set_indexing` unconditionally raises a clear
error whenever a bare identifier inside a `{}` slot isn't itself the target set's own id (i.e. isn't
one of the current-position / shift / explicit-integer forms) — steering authors toward those instead.
There is no follow-up work item for this one, unlike some other restrictions in this plan: the
restriction is permanent by design for every set, since a `Model`'s stored, shared AST can never hold a
per-study- or per-component-specific resolution regardless of what mechanism might be added later.

**Breaking change contained in this PR:** `Model.objective_contributions`/`Model.extra_outputs` change
from `Dict[str, ExpressionNode]` to `Dict[str, IndexedExpression]` (a two-field wrapper carrying
`expression` and `indexed_by`), so an objective-contribution or extra-output can carry its own
`indexed-by`. This is exactly the change flagged in cross-validation finding #4 — the two known-broken
call sites in PR #4's territory are listed there.

**Testing:** `tests/unittests/expressions/visitor/test_indexing.py` — `IndexingStructure` union is
order-independent for equality; `sum_over` collapses only its own dimension; a cross-product structural
check (a term that's both time- and set-indexed reports both). New
`tests/unittests/model/test_set_resolution.py` — one test per classification case in the resolution
pass (current-element drop, shift, explicit literal, standalone bare set-id, unknown-identifier error),
plus a multi-set case with mixed forms (`X{segment, fuel+1}`). Named-element access
(`X{gas}`-style) gets its own dedicated cases, run against **both** a local and a global set to prove
the restriction is uniform: assert `resolve_set_indexing` raises in both, including a case where the
target set's `system.yml` instantiation, if it were resolved early for some other reason, happens to
contain that exact element name — to prove the restriction is structural/permanent for every set and
not merely "not yet resolved" in this particular test's setup. Extend `test_model.py` for the
`IndexedExpression` round-trip and the port-field-definition consistency check.

**Files touched:** new `src/gems/model/set_resolution.py`; modify `src/gems/expression/expression.py`,
`indexing_structure.py`, `indexing.py`; `src/gems/model/{constraint,model,resolve_library,port}.py`;
`src/gems/expression/{copy,print,degree,equality,uses_sum_connections_on,evaluate}.py` (upgrade PR #1's
stubs to real behavior for the 3 new permanent node types); test files listed above.

---

## PR #4 — Data I/O and solver vectorization for global (uniform) sets

**Goal:** a global-set-indexed model can actually be solved end-to-end — data loads, a linopy problem
gets built, results come back with the new `set_id`/`set_index` columns. Local (ragged) sets are
explicitly out of scope here (PR #5).

**Why global sets are the easy 80%:** tracing the actual variable/parameter-array construction code
(`optimization.py`'s `_create_variables_for_model`/`_build_param_arrays_for_model`) confirms GemsPy
builds exactly **one linopy `Variable`/param array per `(model, name)`, shared across every component
of that model**, with dims like `[component, time?, scenario?]`. A global set — required uniform
across every component by PR #2/#3's port-crossing validation — slots into this exact same rectangular
pattern as one more coordinate dimension, essentially risk-free. A **local** set cannot, since
different components can have different cardinality/elements (see PR #5).

**A second de-risking finding:** "cross-product unfolding" needs **no new looping logic** anywhere.
Constraint/objective building today has no explicit loop over `(time, scenario)` at all — unfolding is
entirely implicit in xarray/linopy broadcasting by dimension *name*. The only genuinely 3-dims-hardcoded
places are the `dims`/`coords` construction in `_build_param_arrays_for_model`/
`_create_variables_for_model`, and `simulation_table.py`. Everything else (constraint/objective
`visit`, bound broadcasting, port incidence-matrix multiplication) is already dimension-count-agnostic.
So: extend those two array-construction functions to append `structure.sets` dims with global set
coordinates (identical across every component, unlike time/scenario which are already study-global
anyway); nothing downstream needs new iteration logic.

**Data-series loader (`src/gems/study/data.py`):** `build_data_base` needs a new way to know a
parameter is set-indexed — today it has zero access to `Model`/`Parameter` metadata, only
`ComponentParameterSchema`. Thread through the resolved `System` (or a `Dict[str, Model]` by component
id), already available by this point in `folder.py::load_study`'s call order, so it can check
`component.model.parameters[id].indexed_by`. New `SetIndexedSeriesData` type wrapping an `xr.DataArray`
whose dims are named directly by set ids (plus optional `time`/`scenario`); a parallel
`DataBase.get_set_indexed_values(...)` accessor (the existing scalar `get_value(timestep, scenario)`
contract structurally can't return a multi-dim result). New loader: read the CSV with a header (the
only headered data-series format), validate the column set matches the parameter's declared
`indexed_by` + time/scenario flags exactly, type/range-check each set column (int-in-range for ordinal,
membership-in-`elements` for enumerated), pivot tidy→dense via `pandas.Series.to_xarray()`, and
**explicitly reject any resulting NaN** (an incomplete Cartesian product) with an error naming the
missing combinations — silent NaN leaking into a solver bound would otherwise be a very hard bug to
diagnose. Dispatch is a pure `if param.indexed_by: new loader else: existing loader` branch — the
existing three headerless formats and their code paths are untouched, matching the doc's explicit
backward-compatibility guarantee.

**`sum_over` (`vectorized_builder.py`):** implement `sum_over(self, node: SumOverNode)` (see naming
note above) directly parallel to the existing `all_time_sum`: `.sum(node.set_id)` if the operand
carries that dimension, matching how time/scenario aggregation already works — both
`VectorizedLinearExprBuilder` and `VectorizedExtraOutputBuilder` get this for free from the shared base
`VectorizedBuilderBase`, no override needed, exactly like `all_time_sum`/`scenario_operator` today.
Nested `sum_over` needs no special-casing — recursive `visit` already handles it. Implement
`set_position`/`set_shift`/`set_explicit_index` as the structural analogues of however
`time_shift`/`time_eval` are implemented today (per cross-validation finding #2) — this should not
require inventing a new pattern.

**Simulation table (`simulation_table.py`):** `_da_to_df` currently hardcodes a 3-axis
`transpose("component","time","scenario")` and a hand-rolled `np.repeat`/`np.tile` index construction.
Generalize the transpose to `("component", *set_ids, "time", "scenario")` and replace the manual
row-index construction with `np.indices(da.shape).reshape(...)`, which is both more general *and*
simpler than the current 3-axis-specific code. Add `SimulationColumns.SET_ID`/`SET_INDEX` (inserted
between `scenario_index` and `value`, per the doc's example), constant-per-output `set_id` (blank, or
comma-joined `indexed_by` order) and per-row `set_index` (comma-joined coordinate values, ordinal cast
to int, enumerated kept as the element string). Callers (`_collect_vars_outputs`,
`_collect_extra_outputs`) supply `set_ids` from `var.indexed_by` / the extra-output's `IndexedExpression.indexed_by`.
Also fix the two `.items()` call sites broken by PR #3's `IndexedExpression` wrapper (add `.expression`).

**Port-field aggregation:** tracing `_build_slave_port_array` shows the incidence-matrix multiply
`(A * expr_master_r).sum("component_master")` already broadcasts over any extra dims by name — no code
change needed for `sum_connections` over a global-set-indexed field, since global sets are uniform by
construction. Add one defensive assertion in `build_port_arrays` that the emitted expression's dims
actually match the port field's declared `indexed_by` (fail loudly here rather than deep inside
linopy), and a test locking in the broadcasting behavior.

**Testing:** tidy-CSV parsing tests (single-set ordinal/enumerated, multi-set, full sets+time+scenario,
every error case) in `test_data.py`; `sum_over`/`set_position`/`set_shift`/`set_explicit_index` visitor
tests in `test_vectorized_linear_expr_builder.py`; a new `test_optimization_custom_sets.py` (no
dedicated `optimization.py` unit-test file exists today — only e2e coverage) building a small
synthetic model via the Python construction API directly (not YAML, so this doesn't block on PR #1-#3's
YAML-facing pieces landing in a specific sub-order) with a global set, solving it, and checking values
against hand-computed expectations; a two-component `sum_connections` port test; simulation-table tests
for the new columns including a non-alphabetical declared-order case (catching an implementation that
naively sorts dims instead of respecting `indexed_by` order); one new e2e fixture study (after PR #1-#3
land) producing a real `simulation_table--*.csv` with `set_id`/`set_index` populated.

**Files touched:** `src/gems/study/data.py`; `src/gems/study/resolve_components.py` (thread `System`
through); `src/gems/simulation/{optimization,vectorized_builder,simulation_table}.py`; new
`tests/unittests/simulation/test_optimization_custom_sets.py`; e2e fixture.

---

## PR #5 — Local (ragged) custom sets (recommended follow-up, can ship separately)

**Goal:** a local set whose cardinality or elements genuinely differ per component (not deferred-but-
uniform, like global sets, but actually different values per instance) can be solved too.

**The core problem:** GemsPy's one-array-per-`(model,name)` construction (PR #4) assumes a rectangular
shape shared across every component. A local set breaks that by design — component A might have 3
segments, component B 5. This cannot become "just another array axis" without either padding to a
shared shape, or real per-component variable/constraint construction (a much larger blast radius,
touching every consumer of the shared arrays: bounds, constraints, port arrays, simulation-table
extraction, extra-outputs).

**Recommended strategy: pad to max cardinality across the model's components, plus an explicit
validity mask** — reusing the *existing* `ShiftValidityVisitor`/validity-mask machinery already built
for `OutOfBoundsMode.DROP` on time shifts, generalized from `(component, time)` to `(component,
local_set_dim)`. Padded `(component, local_set_position)` slots get bounds fixed to `0`; any `sum_over`
or constraint touching the ragged dimension must exclude padded slots before aggregating.

**Recommendation: ship PR #1-#4 first.** Global sets alone already cover a large share of realistic use
cases (the doc's own "recommended practice" even steers authors toward universal/global sets with
zero-bound-based per-component variation instead of genuinely ragged local sets, specifically because
of this exact difficulty). Treat this PR as a fast-follow once PR #4 is proven out in real use, not a
blocker for the initial feature launch.

**Files likely touched:** `src/gems/simulation/optimization.py` (padded array construction),
`src/gems/simulation/vectorized_builder.py` (validity-mask-aware `sum_over` and constraint
broadcasting) — exact scope to be re-assessed once PR #4 is merged and real usage patterns are known.
