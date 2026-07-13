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
Tests that integer_strategy is correctly reflected in the built linopy model.

- EXACT    → integer variables keep their type; continuous variables unaffected
- RELAXED  → integer variables become continuous; continuous variables unaffected
- HEURISTIC → same as RELAXED for the solver
"""

from gems.expression.expression import literal
from gems.model.model import model
from gems.model.variable import float_variable, int_variable
from gems.simulation import TimeBlock, build_problem
from gems.study import DataBase, Study, System
from gems.study.parsing import IntegerStrategy, IntegerStrategyId
from gems.study.system import Component

MIXED_MODEL = model(
    id="mixed_model",
    variables=[
        int_variable("nb_units", lower_bound=literal(0), upper_bound=literal(10)),
        float_variable("generation", lower_bound=literal(0)),
    ],
)


def _build(strategies: list[IntegerStrategyId]) -> build_problem:  # type: ignore[valid-type]
    system = System("test")
    for i, strategy in enumerate(strategies):
        component = Component(
            model=MIXED_MODEL,
            id=f"c{i+1}",
            integer_strategy=IntegerStrategy(id=strategy),
        )
        system.add_component(component)
    return build_problem(Study(system, DataBase()), TimeBlock(1, [0]), scenario_ids=[0])


def test_exact_strategy_keeps_integer_variable() -> None:
    problem = _build([IntegerStrategyId.EXACT, IntegerStrategyId.EXACT])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_relaxed_strategy_makes_integer_variable_continuous() -> None:
    problem = _build([IntegerStrategyId.RELAXED, IntegerStrategyId.RELAXED])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_heuristic_strategy_makes_integer_variable_continuous() -> None:
    problem = _build([IntegerStrategyId.HEURISTIC, IntegerStrategyId.HEURISTIC])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_mixed_strategies_work() -> None:
    """Components with different strategies in the same model are supported."""
    problem = _build([IntegerStrategyId.EXACT, IntegerStrategyId.RELAXED])
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )

    problem2 = _build([IntegerStrategyId.HEURISTIC, IntegerStrategyId.EXACT])
    assert (
        problem2.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )
    assert (
        problem2.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem2.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )
