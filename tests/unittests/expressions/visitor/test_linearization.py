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

from unittest.mock import Mock

import pytest

from gems.expression import ExpressionNode, LiteralNode, literal, var
from gems.expression.expression import (
    ComponentVariableNode,
    CurrentScenarioIndex,
    TimeShift,
    comp_param,
    comp_var,
    problem_var,
)
from gems.expression.indexing import IndexingStructureProvider
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.operators_expansion import (
    ProblemDimensions,
    ProblemIndex,
    expand_operators,
)
from gems.simulation.linear_expression import LinearExpression, Term
from gems.simulation.linearize import ParameterGetter, linearize_expression


class AllTimeScenarioDependent(IndexingStructureProvider):
    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)


P = comp_param("c", "p")
X = comp_var("c", "x")
Y = comp_var("c", "y")


def var_at(var: ComponentVariableNode, timestep, scenario) -> LinearExpression:
    return LinearExpression(
        terms=[
            Term(
                1,
                var.component_id,
                var.name,
                time_index=timestep,
                scenario_index=scenario,
            )
        ],
        constant=0,
    )


def X_at(t: int = 0, s: int = 0) -> LinearExpression:
    return var_at(X, timestep=t, scenario=s)


def Y_at(t: int = 0, s: int = 0) -> LinearExpression:
    return var_at(Y, timestep=t, scenario=s)


def constant(c: float) -> LinearExpression:
    return LinearExpression([], c)


def evaluate_literal(node: ExpressionNode) -> int:
    if isinstance(node, LiteralNode):
        return int(node.value)
    raise NotImplementedError("Can only evaluate literal nodes.")


def test_linearization_before_operator_substitution_raises_an_error() -> None:
    x = var("x")
    expr = x.variance()

    with pytest.raises(
        ValueError, match="Scenario operators need to be expanded before linearization"
    ):
        linearize_expression(expr, timestep=0, scenario=0)


def _expand_and_linearize(
    expr: ExpressionNode,
    dimensions: ProblemDimensions,
    index: ProblemIndex,
    parameter_value_provider: ParameterGetter,
) -> LinearExpression:
    expanded = expand_operators(
        expr, dimensions, evaluate_literal, AllTimeScenarioDependent()
    )
    return linearize_expression(
        expanded, index.timestep, index.scenario, parameter_value_provider
    )


@pytest.mark.parametrize(
    "expr,expected",
    [
        ((5 * X + 3) / 2, constant(2.5) * X_at(t=0) + constant(1.5)),
        ((X + Y).time_sum(), X_at(t=0) + Y_at(t=0) + X_at(t=1) + Y_at(t=1)),
        (X.shift(-1).shift(+1), X_at(t=0)),
        (X.shift(-1).time_sum(), X_at(t=-1) + X_at(t=0)),
        (X.shift(-1).time_sum(-1, +1), X_at(t=-2) + X_at(t=-1) + X_at(t=0)),
        (X.time_sum().shift(-1), X_at(t=-1) + X_at(t=0)),
        (X.time_sum(-1, +1).shift(-1), X_at(t=-2) + X_at(t=-1) + X_at(t=0)),
        (X.eval(2).time_sum(), X_at(t=2) + X_at(t=2)),
        ((X + 2).time_sum(), X_at(t=0) + X_at(t=1) + constant(4)),
        ((X + 2).time_sum(-1, 0), X_at(t=-1) + X_at(t=0) + constant(4)),
        ((X + 2).time_sum(-1, 0), X_at(t=-1) + X_at(t=0) + constant(4)),
    ],
)
def test_linearization_of_nested_time_operations(
    expr: ExpressionNode, expected: LinearExpression
) -> None:
    dimensions = ProblemDimensions(timesteps_count=2, scenarios_count=1)
    index = ProblemIndex(timestep=0, scenario=0)
    params = Mock(spec=ParameterGetter)

    assert _expand_and_linearize(expr, dimensions, index, params) == expected


def test_invalid_multiplication() -> None:
    params = Mock(spec=ParameterGetter)

    x = problem_var(
        "c", "x", time_index=TimeShift(0), scenario_index=CurrentScenarioIndex()
    )
    expression = x * x
    with pytest.raises(ValueError, match="constant"):
        linearize_expression(expression, 0, 0, params)


def test_invalid_division() -> None:
    params = Mock(spec=ParameterGetter)

    x = problem_var(
        "c", "x", time_index=TimeShift(0), scenario_index=CurrentScenarioIndex()
    )
    expression = literal(1) / x
    with pytest.raises(ValueError, match="constant"):
        linearize_expression(expression, 0, 0, params)
