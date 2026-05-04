# GemsPy Changelog

All notable changes to GemsPy are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

---

## v0.1.0 — Initial Release

### Added

- Pure-Python interpreter for the GEMS modelling language
- ANTLR4-based expression parser with full mathematical syntax support (`grammar/Expr.g4`)
- Model and library loading from YAML (`model/`, `libs/`)
- Study instantiation from folder structure (`study/`)
- Optimization problem construction via [linopy](https://github.com/PyPSA/linopy) with HiGHS solver backend (`simulation/`)
- Temporal decomposition support: sequential subproblems, parallel subproblems, and Benders decomposition (`optim_config/`)
- CLI entry point: `gemspy --model-libs ... --components ... --timeseries ... --duration ... --scenarios ...`
- Python API: `load_study()`, `run_study()`, programmatic `Study` + `build_problem()` interface
- Support for Python 3.10, 3.11, 3.12
- CI pipeline: linting (`black`, `isort`), type checking (`mypy`), tests with coverage (`pytest --cov`)
- PyPI package: `gemspy`
