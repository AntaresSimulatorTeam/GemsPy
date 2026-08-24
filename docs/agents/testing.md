# Testing Strategy

## Philosophy

Tests live alongside the module they exercise. Fixtures (YAML snippets, small networks) are kept
in `tests/` sub-directories. No mocking of the solver—tests use real HiGHS calls via linopy
(`problem.solve(solver_name="highs")`).

`tests/unittests/` is organized **by package first** (`gems_craft/`, `gems_craft_hybrid/`,
`gems_runner/`, mirroring `src/`), then by topic underneath — so it's always clear which package
a unit test exercises. `tests/e2e/` is *not* split by package: those tests are cross-cutting by
nature (they build a study with `gems_craft` and solve it with `gems_runner` in the same test).

## Layers

| Layer | Location | Description |
|---|---|---|
| Unit — gems_craft / data | `tests/unittests/gems_craft/data/` | `DataBase` and data resolution |
| Unit — gems_craft / expressions | `tests/unittests/gems_craft/expressions/` | AST visitors and expression parsing (parsing/, visitor/) |
| Unit — gems_craft / libraries | `tests/unittests/gems_craft/lib_parsing/` | Model library YAML parsing |
| Unit — gems_craft / optim config | `tests/unittests/gems_craft/optim_config/` | Optimization-config YAML parsing |
| Unit — gems_craft / scenario builder | `tests/unittests/gems_craft/scenario_builder/` | Scenario and time-series builder |
| Unit — gems_craft / system | `tests/unittests/gems_craft/system/` | Model, network, and port object behaviour |
| Unit — gems_craft / system parsing | `tests/unittests/gems_craft/system_parsing/` | System YAML parsing |
| Unit — gems_craft_hybrid | `tests/unittests/gems_craft_hybrid/` | Hybrid GEMS/Antares Simulator schema parsing |
| Unit — gems_runner / expression | `tests/unittests/gems_runner/expression/` | Solver-output expression evaluation (`dual()`, `reduced_cost()`, `lower_bound()`, `upper_bound()`, `variable()`) |
| Unit — gems_runner / simulation | `tests/unittests/gems_runner/simulation/` | Full problem build + solve on small networks |
| End-to-end — functional | `tests/e2e/functional/` | Cross-cutting tests: library/system combinations, stochastic, investment, scenario builder |
| End-to-end — models | `tests/e2e/models/` | Model-level tests (andromede-v1 models, operator tests, proof-of-concept models) |
| End-to-end — studies | `tests/e2e/functional/studies/` | Full YAML study fixtures read by the functional end-to-end tests |
