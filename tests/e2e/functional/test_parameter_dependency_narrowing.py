# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
"""End-to-end tests for per-component parameter dependency narrowing.

A model parameter declares the *maximum* dependency (time, scenario); a
component may declare the same or less.  These tests solve a small system for
each allowed combination and check that the data actually used by the solver
follows the *component's* declaration: the parameter is constant along every
axis the component declared independent, and the objective value reflects that.
"""

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base, resolve_system
from gems_craft.study.study import Study
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.optimization import OptimizationProblem

_LIBRARY = """\
library:
  id: basic
  port-types:
    - id: flow
      fields: [ {id: flow} ]
  models:
    - id: node
      ports: [ {id: p, type: flow} ]
      binding-constraints:
        - id: balance
          expression: sum_connections(p.flow) = 0
    - id: gen
      parameters:
        - id: cost
          time-dependent: %(model_time)s
          scenario-dependent: %(model_scenario)s
      variables:
        - id: g
          lower-bound: 0
          upper-bound: 1000
      ports: [ {id: p, type: flow} ]
      port-field-definitions:
        - port: p
          field: flow
          definition: g
      objective-contributions:
        - id: operational
          expression: expec(sum(cost * g))
    - id: load
      parameters:
        - id: d
          time-dependent: true
          scenario-dependent: true
      ports: [ {id: p, type: flow} ]
      port-field-definitions:
        - port: p
          field: flow
          definition: -d
"""

_SYSTEM = """\
system:
  id: narrowing
  components:
    - id: N
      model: basic.node
    - id: G
      model: basic.gen
      parameters:
        - id: cost
          time-dependent: %(component_time)s
          scenario-dependent: %(component_scenario)s
          value: %(cost_value)s
    - id: D
      model: basic.load
      parameters:
        - id: d
          time-dependent: true
          scenario-dependent: true
          value: demand
  connections:
    - component1: G
      port1: p
      component2: N
      port2: p
    - component1: D
      port1: p
      component2: N
      port2: p
"""

# Demand is time x scenario: rows are timesteps, columns are scenarios.
_DEMAND = "10 20\n30 40\n"

# Cost series, one file per shape the "cost" parameter may take.
_COST_SERIES = {
    "cost_time": "5\n7\n",  # T x 1
    "cost_scenario": "5 7\n",  # 1 x S
    "cost_time_scenario": "5 7\n9 11\n",  # T x S
}


@pytest.fixture
def series_files(tmp_path: Path) -> Path:
    """Write the data-series files used by the tests, return their directory."""
    (tmp_path / "demand.txt").write_text(_DEMAND)
    for name, content in _COST_SERIES.items():
        (tmp_path / f"{name}.txt").write_text(content)
    return tmp_path


def _build(
    series_dir: Path,
    model_dependency: Tuple[bool, bool],
    component_dependency: Tuple[bool, bool],
    cost_value: str,
) -> OptimizationProblem:
    """Resolve and build the problem for one dependency combination."""

    def yaml_bool(value: bool) -> str:
        return "true" if value else "false"

    library = parse_yaml_library(
        io.StringIO(
            _LIBRARY
            % {
                "model_time": yaml_bool(model_dependency[0]),
                "model_scenario": yaml_bool(model_dependency[1]),
            }
        )
    )
    input_system = parse_yaml_system(
        io.StringIO(
            _SYSTEM
            % {
                "component_time": yaml_bool(component_dependency[0]),
                "component_scenario": yaml_bool(component_dependency[1]),
                "cost_value": cost_value,
            }
        )
    )
    system = resolve_system(input_system, resolve_library([library]))
    database = build_data_base(input_system, series_dir)
    return build_problem(Study(system, database), TimeBlock(1, [0, 1]), [0, 1])


def _cost_values(problem: OptimizationProblem) -> np.ndarray:
    """The cost data as the solver sees it, broadcast to dims (time, scenario).

    The demand parameter is always time- and scenario-dependent, so it supplies
    the full (time, scenario) grid to broadcast the cost against.
    """
    cost = problem.param_arrays[("basic.gen", "cost")].sel(component="G")
    grid = problem.param_arrays[("basic.load", "d")].sel(component="D")
    return np.asarray(cost.broadcast_like(grid).transpose("time", "scenario").values)


# ---------------------------------------------------------------------------
# The component's declaration prevails over the model's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_dependency, component_dependency, cost_value, expected_objective",
    [
        # Model declares both axes; the component narrows to fewer and fewer.
        ((True, True), (True, True), "cost_time_scenario", 450.0),
        ((True, True), (True, False), "cost_time", 320.0),
        ((True, True), (False, True), "cost_scenario", 310.0),
        ((True, True), (False, False), "3", 150.0),
        # Model declares one axis; the component keeps it, then drops it.
        ((False, True), (False, True), "cost_scenario", 310.0),
        ((False, True), (False, False), "3", 150.0),
        ((True, False), (True, False), "cost_time", 320.0),
        ((True, False), (False, False), "3", 150.0),
    ],
)
def test_component_dependency_prevails_over_model_dependency(
    series_files: Path,
    model_dependency: Tuple[bool, bool],
    component_dependency: Tuple[bool, bool],
    cost_value: str,
    expected_objective: float,
) -> None:
    """Data varies only along the axes the *component* declares.

    The generator must cover the demand exactly, so the objective is the
    scenario-average of ``sum_t(cost[t, s] * demand[t, s])``.  Each expected
    value below is only reachable if ``cost`` is broadcast over the axes the
    component declared independent.
    """
    problem = _build(series_files, model_dependency, component_dependency, cost_value)
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(expected_objective)

    cost = _cost_values(problem)  # dims (time, scenario)
    component_time, component_scenario = component_dependency

    if not component_time:
        assert np.all(cost == cost[0, :][np.newaxis, :]), "cost varies along time"
    if not component_scenario:
        assert np.all(cost == cost[:, 0][:, np.newaxis]), "cost varies along scenario"
    if component_time:
        assert cost[0, 0] != cost[1, 0], "cost is expected to vary along time"
    if component_scenario:
        assert cost[0, 0] != cost[0, 1], "cost is expected to vary along scenario"


def test_narrowing_does_not_change_the_problem_shape(series_files: Path) -> None:
    """Narrowing changes values, not the shape of the optimization problem.

    The parameter array keeps the dimensions declared by the *model*, so a
    narrowed component yields the same variables and constraints as a
    non-narrowed one — only the numbers differ.
    """
    full = _build(series_files, (True, True), (True, True), "cost_time_scenario")
    narrowed = _build(series_files, (True, True), (False, False), "3")

    assert (
        full.param_arrays[("basic.gen", "cost")].dims
        == narrowed.param_arrays[("basic.gen", "cost")].dims
    )
    assert len(full.linopy_model.variables) == len(narrowed.linopy_model.variables)
    assert len(full.linopy_model.constraints) == len(narrowed.linopy_model.constraints)
