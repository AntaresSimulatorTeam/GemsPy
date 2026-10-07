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

"""Constant terms in objective contributions (issue #254)."""

from typing import Dict

import pytest

from gems_craft.expression.expression import ExpressionNode, literal, var
from gems_craft.model import Constraint, float_variable, model
from gems_craft.study import Component, DataBase, Study, System
from gems_runner.simulation import TimeBlock, build_problem


@pytest.mark.parametrize(
    "contributions, expected",
    [
        pytest.param(
            {"obj": (3 * var("x") + 5).time_sum().expec()}, 3 * 2 + 5, id="mixed"
        ),
        pytest.param({"obj": literal(5)}, 5, id="constant_only"),
        pytest.param(
            {"obj": (3 * var("x")).time_sum().expec(), "fixed": literal(5)},
            3 * 2 + 5,
            id="separate_terms",
        ),
    ],
)
def test_constant_in_objective(
    contributions: Dict[str, ExpressionNode], expected: float
) -> None:
    m = model(
        id="M",
        variables=[float_variable("x", lower_bound=literal(0))],
        constraints=[Constraint(name="min_x", expression=var("x") >= literal(2))],
        objective_contributions=contributions,
    )
    system = System("test")
    system.add_component(Component(model=m, id="C"))

    problem = build_problem(Study(system, DataBase()), TimeBlock(1, [0]), [0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(expected)
