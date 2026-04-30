# Optimisation: building and solving problems with GemsPy

GemsPy offers three levels of abstraction for running an optimisation:

| Level | Entry point | Best for |
|---|---|---|
| High-level (recommended) | `run_study()` | Directory-based studies, no custom logic |
| Mid-level | `SimulationSession` | Full control over config and resolution mode |
| Low-level | `build_problem()` | Fine-grained programmatic control |

---

## High-level: `run_study()`

The simplest way to run a study is a single function call.  It reads the study
directory, loads `input/optim-config.yml` (using defaults when absent), solves
the problem, and writes results to `output/<run_id>/`.

~~~ python
from pathlib import Path
from gems.study.runner import run_study

run_study(Path("my_study"))
~~~

An alternative `optim-config.yml` path can be provided:

~~~ python
run_study(Path("my_study"), optim_config_path=Path("configs/my_config.yml"))
~~~

---

## Mid-level: `SimulationSession`

`SimulationSession` gives you explicit control over which configuration is used
and lets you inspect the returned `SimulationTable` in memory.

~~~ python
from pathlib import Path
from gems.study.folder import load_study
from gems.session import SimulationSession
from gems.optim_config import load_optim_config

study = load_study(Path("my_study"))
optim_config = load_optim_config(Path("my_study/input/optim-config.yml"))

session = SimulationSession(study=study, optim_config=optim_config)
results = session.run()  # returns a SimulationTable
~~~

The resolution strategy (frontal, sequential, parallel, Benders) is selected
automatically from `optim_config.resolution.mode`.  See
[Optimisation configuration](optim-config.md) for details.

---

## Low-level: `build_problem()`

Use `build_problem()` when you need direct access to the `OptimizationProblem`
object — for example to export an LP file, inspect variable labels, or chain
multiple solve calls.

### Building the optimisation problem

~~~ python
from gems.study import Study, load_study
from gems.simulation import build_problem, TimeBlock

study = load_study(Path("my_study"))

problem = build_problem(
    study,
    TimeBlock(1, list(range(timespan))),
    scenario_ids=list(range(nb_scenarios)),
)
~~~

`scenario_ids` is a list of Monte-Carlo scenario indices (0-based).  For
example, `scenario_ids=[0, 1, 2]` runs three scenarios.

### Solving the optimisation problem

~~~ python
problem.solve()                    # uses HiGHS by default
print(problem.objective_value)     # float
print(problem.status)              # 'ok' or 'warning'
~~~

A different solver or additional HiGHS options can be passed:

~~~ python
problem.solve(solver_name="highs", threads=4)
~~~

### Exporting the LP file

~~~ python
problem.export_lp(Path("debug.lp"))
~~~
