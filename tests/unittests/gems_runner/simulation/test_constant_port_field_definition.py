# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
Tests that a port field defined by a constant expression (e.g. ``definition: 0``)
builds and solves, the constant being contributed once per connection.
"""

import io
from pathlib import Path

import pytest

from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study import Study
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base, resolve_system
from gems_runner.simulation import TimeBlock, build_problem

DEMAND = 10.0
NB_TIMESTEPS = 3

LIBRARY = """
library:
  id: lib
  port-types:
    - id: flow
      fields:
        - id: flow
        - id: injection
        - id: bound
  models:
    - id: node
      ports:
        - id: balance_port
          type: flow
      binding-constraints:
        - id: balance
          expression: sum_connections(balance_port.flow) + sum_connections(balance_port.injection) = 0
    - id: generator
      parameters:
        - id: demand
          time-dependent: false
          scenario-dependent: false
      variables:
        - id: generation
          lower-bound: 0
      ports:
        - id: balance_port
          type: flow
      port-field-definitions:
        - port: balance_port
          field: flow
          definition: generation - demand
        - port: balance_port
          field: injection
          definition: {injection}
        - port: balance_port
          field: bound
          definition: {bound}
      objective-contributions:
        - id: objective
          expression: sum(generation)
"""


def _system(nb_generators: int) -> str:
    components = "\n".join(f"""    - id: gen{i}
      model: lib.generator
      parameters:
        - id: demand
          time-dependent: false
          scenario-dependent: false
          value: {DEMAND}""" for i in range(nb_generators))
    connections = "\n".join(f"""    - component1: node
      port1: balance_port
      component2: gen{i}
      port2: balance_port""" for i in range(nb_generators))
    return f"""
system:
  id: system
  components:
    - id: node
      model: lib.node
{components}
  connections:
{connections}
"""


def _solve(injection: str, bound: str, nb_generators: int, tmp_path: Path) -> float:
    library = parse_yaml_library(
        io.StringIO(LIBRARY.format(injection=injection, bound=bound))
    )
    system_schema = parse_yaml_system(io.StringIO(_system(nb_generators)))
    study = Study(
        resolve_system(system_schema, resolve_library([library])),
        build_data_base(system_schema, tmp_path),
    )
    problem = build_problem(study, TimeBlock(0, list(range(NB_TIMESTEPS))), [0])
    problem.solve(solver_name="highs")
    return problem.linopy_model.objective.value


@pytest.mark.parametrize("nb_generators", [1, 2])
@pytest.mark.parametrize("injection, value", [("0", 0.0), ("1.5", 1.5), ("2 * 3", 6.0)])
def test_constant_port_field_is_contributed_once_per_connection(
    injection: str, value: float, nb_generators: int, tmp_path: Path
) -> None:
    objective = _solve(injection, "0 * generation", nb_generators, tmp_path)

    assert objective == pytest.approx(nb_generators * (DEMAND - value) * NB_TIMESTEPS)
    assert objective == pytest.approx(
        _solve(
            f"{injection} + 0 * generation", "0 * generation", nb_generators, tmp_path
        )
    )


@pytest.mark.parametrize("nb_generators", [1, 2])
def test_constant_port_field_not_read_by_receiver(
    nb_generators: int, tmp_path: Path
) -> None:
    objective = _solve("0 * generation", "0", nb_generators, tmp_path)

    assert objective == pytest.approx(nb_generators * DEMAND * NB_TIMESTEPS)
