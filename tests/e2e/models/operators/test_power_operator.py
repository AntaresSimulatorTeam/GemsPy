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

"""
End-to-end coverage of the power operator '^' driven from YAML.

Exercises '^' in a variable bound, a constraint, an objective contribution
(parameters and literals only) and in an extra-output (applied to a decision
variable, which is only legal post-solve).
"""

import io
import math
from pathlib import Path

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study import Study
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base, resolve_system
from gems_runner.simulation import build_problem
from gems_runner.simulation.simulation_table import SimulationTableBuilder
from gems_runner.simulation.time_block import TimeBlock

LIBRARY_YAML = """
library:
  id: power_lib
  description: Exercises the '^' operator in every expression context.
  port-types: []
  models:
    - id: powered
      parameters:
        - id: p
        - id: cost
      variables:
        - id: gen
          lower-bound: 0
          upper-bound: p^3
          variable-type: continuous
      constraints:
        - id: fix_generation
          expression: gen = p^2
      objective-contributions:
        - id: objective
          expression: expec(sum(cost^2 * gen))
      extra-outputs:
        - id: squared
          expression: gen^2
        - id: two_to_the_p
          expression: 2^p
        - id: precedence
          expression: gen + -2^2
"""

SYSTEM_YAML = """
system:
  components:
  - id: unit
    model: power_lib.powered
    parameters:
    - id: p
      scenario-dependent: false
      time-dependent: false
      value: 3.0
    - id: cost
      scenario-dependent: false
      time-dependent: false
      value: 2.0
"""

# p = 3, cost = 2, over 2 timesteps:
#   gen  = p^2       = 9      (upper bound p^3 = 27 is not binding)
#   obj  = cost^2 * gen * 2 timesteps = 4 * 9 * 2 = 72
EXPECTED_GENERATION = 9.0
EXPECTED_OBJECTIVE = 72.0


@pytest.fixture
def study(tmp_path: Path) -> Study:
    library = resolve_library([parse_yaml_library(io.StringIO(LIBRARY_YAML))])
    system = resolve_system(parse_yaml_system(io.StringIO(SYSTEM_YAML)), library)
    database = build_data_base(parse_yaml_system(io.StringIO(SYSTEM_YAML)), tmp_path)
    return Study(system, database)


def test_power_operator_end_to_end(study: Study) -> None:
    problem = build_problem(study, TimeBlock(1, [0, 1]), list(range(1)))
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert math.isclose(problem.objective_value, EXPECTED_OBJECTIVE, rel_tol=1e-6)

    df = SimulationTableBuilder().build(problem)

    def value(output: str, time_index: int = 0) -> float:
        return (
            df.component("unit")
            .output(output)
            .value(time_index=time_index, scenario_index=0)
        )

    assert value("gen") == pytest.approx(EXPECTED_GENERATION)
    # '^' applied to a decision variable: legal in an extra-output only.
    assert value("squared") == pytest.approx(81.0)
    # '^' with a parameter exponent.
    assert value("two_to_the_p") == pytest.approx(8.0)
    # '^' binds tighter than unary minus: -2^2 is -(2^2) = -4, not (-2)^2 = 4,
    # so this is 9 - 4 and not 9 + 4.
    assert value("precedence") == pytest.approx(5.0)
