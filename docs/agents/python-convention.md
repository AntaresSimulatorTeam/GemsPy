# Python conventions

## Code Style & Conventions

- **Formatter**: Black, line-length 88. Never adjust line breaks manually—let Black decide.
- **Import order**: isort with `profile = "black"`. One `import` block, no manual blank lines between
  groups.
- **Type annotations**: All functions and methods **must** have full type annotations. mypy is run in
  strict mode (`disallow_untyped_defs`, `disallow_untyped_calls`).
- **Dataclasses**: Prefer `@dataclass(frozen=True)` for value objects (expression nodes, model
  definitions). Mutability must be justified explicitly.
- **Pydantic**: Use `ConfigDict(alias_generator=to_camel)` or kebab-case alias generation for YAML
  round-tripping; use Pydantic v2 APIs only.
- **Naming**:
  - Classes: `PascalCase`
  - Functions / variables: `snake_case`
  - Constants / type-level aliases: `UPPER_SNAKE_CASE`
  - YAML keys: `kebab-case`
- **Commit messages**: Follow Conventional Commits — `<type>(<scope>): <summary>`, e.g.
  `feat(expression): add floor operator`. Types used in this repo: `feat`, `fix`, `docs`, `refactor`,
  `test`, `chore`, `release`.

---

## Agent Guardrails

Rules for automated agents (CI bots, AI coding assistants, Dependabot, etc.):

1. **Never auto-merge** to `main`; all changes require at least one human review.
2. **Do not edit generated files** under `src/gems/expression/parsing/antlr/`; regenerate them
   from `grammar/Expr.g4` instead.
3. **Do not modify `pyproject.toml` version** manually; version bumps are handled via the
   `feat(release):` commit workflow.
4. **Keep pre-commit hooks passing**: any commit that breaks `black`, `isort`, or `mypy` must not
   be auto-pushed.
5. **Test coverage must not decrease** on `main`; PRs that drop coverage without justification
   should be flagged.
