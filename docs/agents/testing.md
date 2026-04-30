# Testing Strategy

## Philosophy

Tests live alongside the module they exercise. Fixtures (YAML snippets, small networks) are kept
in `tests/` sub-directories. No mocking of the solver—tests use real HiGHS calls via linopy
(`problem.solve(solver_name="highs")`).

## Layers

| Layer | Location | Description |
|---|---|---|
| Unit — expressions | `tests/unittests/expressions/` | AST visitors and expression parsing (parsing/, visitor/) |
| Unit — libraries | `tests/unittests/lib_parsing/` | Model library YAML parsing |
| Unit — system | `tests/unittests/system/` | Model, network, and port object behaviour |
| Unit — system parsing | `tests/unittests/system_parsing/` | System YAML parsing |
| Unit — scenario builder | `tests/unittests/scenario_builder/` | Scenario and time-series builder |
| Integration | `tests/unittests/simulation/` | Full problem build + solve on small networks |
| End-to-end — functional | `tests/e2e/functional/` | Cross-cutting tests: library/system combinations, stochastic, investment, scenario builder |
| End-to-end — models | `tests/e2e/models/` | Model-level tests (andromede-v1 models, operator tests, proof-of-concept models) |
| End-to-end — studies | `tests/e2e/studies/` | Study-level tests reading full YAML study fixtures |
