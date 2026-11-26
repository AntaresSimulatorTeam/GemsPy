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
Tests that a NaN, or an infinite value where it is not allowed, raises
instead of silently changing the linopy model.
"""

import io
from pathlib import Path

import pytest

from gems_craft.expression.expression import literal, param, port_field, var
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model import (
    Constraint,
    ModelPort,
    PortType,
    float_parameter,
    float_variable,
    model,
)
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.port import PortField, PortFieldDefinition, PortFieldId
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study import (
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    create_component,
)
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base, resolve_system
from gems_runner.simulation import NonFiniteValueError, TimeBlock, build_problem

CONSTANT = IndexingStructure(False, False)

LIBRARY = """
library:
  id: lib
  models:
    - id: m
      parameters:
        - id: p
          time-dependent: false
          scenario-dependent: false
        - id: q
          time-dependent: false
          scenario-dependent: false
      variables:
        - id: x
          lower-bound: 0
          upper-bound: {upper_bound}
      constraints:
        - id: c
          expression: {constraint}
      objective-contributions:
        - id: cost
          expression: {objective}
"""

SYSTEM = """
system:
  id: system
  components:
    - id: ok
      model: lib.m
      parameters:
        - id: p
          time-dependent: false
          scenario-dependent: false
          value: 1
        - id: q
          time-dependent: false
          scenario-dependent: false
          value: 1
    - id: bad
      model: lib.m
      parameters:
        - id: p
          time-dependent: false
          scenario-dependent: false
          value: {p}
        - id: q
          time-dependent: false
          scenario-dependent: false
          value: 0
"""


def _build(
    tmp_path: Path,
    constraint: str = "x <= 10",
    objective: str = "sum(x)",
    upper_bound: str = "10",
    p: float = 0,
) -> Study:
    library = parse_yaml_library(
        io.StringIO(
            LIBRARY.format(
                constraint=constraint, objective=objective, upper_bound=upper_bound
            )
        )
    )
    system_schema = parse_yaml_system(io.StringIO(SYSTEM.format(p=p)))
    return Study(
        resolve_system(system_schema, resolve_library([library])),
        build_data_base(system_schema, tmp_path),
    )


@pytest.mark.parametrize(
    "kwargs, what",
    [
        ({"constraint": "(p / q) * x <= 10"}, "nan in constraint 'c'"),
        ({"constraint": "x <= p^0.5", "p": -1}, "nan in constraint 'c'"),
        ({"constraint": "x >= 1 / q"}, "-inf in constraint 'c'"),
        ({"objective": "sum((1 / q) * x)"}, "inf in objective contribution 'cost'"),
        ({"upper_bound": "p / q"}, "nan in upper bound of variable 'x'"),
    ],
)
def test_invalid_value_raises(kwargs: dict, what: str, tmp_path: Path) -> None:
    study = _build(tmp_path, **kwargs)

    with pytest.raises(
        NonFiniteValueError,
        match=f"Invalid value {what}, model 'lib.m', component 'bad'",
    ):
        build_problem(study, TimeBlock(0, [0, 1]), [0])


def test_infinite_bounds_mean_no_bound(tmp_path: Path) -> None:
    # 1 / q is +inf for component "bad": an upper bound or right-hand side of +inf is no bound.
    study = _build(tmp_path, constraint="x <= 1 / q", upper_bound="1 / q")
    problem = build_problem(study, TimeBlock(0, [0, 1]), [0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"


def test_invalid_value_in_port_field_names_the_port_field() -> None:
    port_type = PortType(id="flow_port", fields=[PortField("flow")])
    producer = model(
        id="PRODUCER",
        parameters=[float_parameter("q", CONSTANT)],
        variables=[float_variable("x", lower_bound=literal(0))],
        ports=[ModelPort(port_type=port_type, port_name="out")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId("out", "flow"),
                definition=(literal(1) / param("q")) * var("x"),
            )
        ],
    )
    node = model(
        id="NODE",
        ports=[ModelPort(port_type=port_type, port_name="in")],
        binding_constraints=[
            Constraint(
                name="balance",
                expression=port_field("in", "flow").sum_connections() == literal(1),
            )
        ],
    )
    database = DataBase()
    database.add_data("P", "q", ConstantData(0))
    system = System("port")
    p = create_component(model=producer, id="P")
    n = create_component(model=node, id="N")
    system.add_component(p)
    system.add_component(n)
    system.connect(PortRef(p, "out"), PortRef(n, "in"))

    with pytest.raises(
        NonFiniteValueError,
        match="Invalid value inf in definition of port field 'out.flow', "
        "model 'PRODUCER', component 'P'",
    ):
        build_problem(Study(system, database), TimeBlock(0, [0, 1]), [0])
