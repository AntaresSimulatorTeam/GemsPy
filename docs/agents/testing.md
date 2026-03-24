# Testing Strategy

## Philosophy

Tests live alongside the module they exercise. Fixtures (YAML snippets, small networks) are kept
in `tests/` sub-directories. No mocking of the solver—tests use real OR-Tools calls.

## Layers

| Layer | Location | Description |
|---|---|---|
| Unit | `tests/unittests/` | One file per module area; covers AST visitors, parsing, model loading, linearisation |
| Integration | `tests/unittests/simulation/` | Full problem build + solve on tiny networks |
| End-to-end | `tests/e2e/` | CLI-level tests reading real YAML fixtures |
| Converter | `tests/input_converter/`, `tests/pypsa_converter/`, `tests/antares_historic/` | Format-specific conversion tests |
