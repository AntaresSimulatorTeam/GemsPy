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

from dataclasses import dataclass, field
from typing import Dict

import pytest

from gems.expression import (
    AdditionNode,
    DivisionNode,
    EvaluationContext,
    EvaluationVisitor,
    ExpressionNode,
    LiteralNode,
    ParameterNode,
    PrinterVisitor,
    ValueProvider,
    VariableNode,
    literal,
    param,
    sum_expressions,
    var,
    visit,
)
from gems.expression.equality import expressions_equal
from gems.expression.expression import ComponentParameterNode, ComponentVariableNode


@dataclass(frozen=True)
class ComponentValueKey:
    component_id: str
    variable_name: str


def comp_key(component_id: str, variable_name: str) -> ComponentValueKey:
    return ComponentValueKey(component_id, variable_name)


@dataclass(frozen=True)
class ComponentEvaluationContext(ValueProvider):
    """
    Simple value provider relying on dictionaries.
    Does not support component variables/parameters.
    """

    variables: Dict[ComponentValueKey, float] = field(default_factory=dict)
    parameters: Dict[ComponentValueKey, float] = field(default_factory=dict)

    def get_variable_value(self, name: str) -> float:
        raise NotImplementedError()

    def get_parameter_value(self, name: str) -> float:
        raise NotImplementedError()

    def get_component_variable_value(self, component_id: str, name: str) -> float:
        return self.variables[comp_key(component_id, name)]

    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        return self.parameters[comp_key(component_id, name)]


def test_comp_parameter() -> None:
    add_node = AdditionNode([LiteralNode(1), ComponentVariableNode("comp1", "x")])
    expr = DivisionNode(add_node, ComponentParameterNode("comp1", "p"))

    assert visit(expr, PrinterVisitor()) == "((1 + comp1.x) / comp1.p)"

    context = ComponentEvaluationContext(
        variables={comp_key("comp1", "x"): 3}, parameters={comp_key("comp1", "p"): 4}
    )
    assert visit(expr, EvaluationVisitor(context)) == 1


def test_ast() -> None:
    add_node = AdditionNode([LiteralNode(1), VariableNode("x")])
    expr = DivisionNode(add_node, ParameterNode("p"))

    assert visit(expr, PrinterVisitor()) == "((1 + x) / p)"

    context = EvaluationContext(variables={"x": 3}, parameters={"p": 4})
    assert visit(expr, EvaluationVisitor(context)) == 1


def test_operators() -> None:
    x = var("x")
    p = param("p")
    expr: ExpressionNode = (5 * x + 3) / p - 2

    assert visit(expr, PrinterVisitor()) == "((((5.0 * x) + 3.0) / p) - 2.0)"

    context = EvaluationContext(variables={"x": 3}, parameters={"p": 4})
    assert visit(expr, EvaluationVisitor(context)) == pytest.approx(2.5, 1e-16)

    assert visit(-expr, EvaluationVisitor(context)) == pytest.approx(-2.5, 1e-16)


def test_sum_expressions() -> None:
    assert expressions_equal(sum_expressions([]), literal(0))
    assert expressions_equal(sum_expressions([literal(1)]), literal(1))
    assert expressions_equal(sum_expressions([literal(1), var("x")]), 1 + var("x"))
    assert expressions_equal(
        sum_expressions([literal(1), var("x"), param("p")]), 1 + (var("x") + param("p"))
    )
