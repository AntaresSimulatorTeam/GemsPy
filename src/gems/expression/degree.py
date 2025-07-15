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

import gems.expression.scenario_operator
from gems.expression.expression import (
    AllTimeSumNode,
    ComponentParameterNode,
    ComponentVariableNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
)

from .expression import (
    AdditionNode,
    ComparisonNode,
    DivisionNode,
    ExpressionNode,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    ScenarioOperatorNode,
    VariableNode,
)
from .visitor import ExpressionVisitor, T, visit


class ExpressionDegreeVisitor(ExpressionVisitor[int]):
    """
    Computes degree of expression with respect to variables.
    """

    def literal(self, node: LiteralNode) -> int:
        return 0

    def negation(self, node: NegationNode) -> int:
        return visit(node.operand, self)

    # TODO: Take into account simplification that can occur with literal coefficient for add, sub, mult, div
    def addition(self, node: AdditionNode) -> int:
        degrees = [visit(o, self) for o in node.operands]
        return max(degrees)

    def multiplication(self, node: MultiplicationNode) -> int:
        return visit(node.left, self) + visit(node.right, self)

    def division(self, node: DivisionNode) -> int:
        right_degree = visit(node.right, self)
        if right_degree != 0:
            raise ValueError("Degree computation not implemented for divisions.")
        return visit(node.left, self)

    def comparison(self, node: ComparisonNode) -> int:
        return max(visit(node.left, self), visit(node.right, self))

    def variable(self, node: VariableNode) -> int:
        return 1

    def parameter(self, node: ParameterNode) -> int:
        return 0

    def comp_variable(self, node: ComponentVariableNode) -> int:
        return 1

    def comp_parameter(self, node: ComponentParameterNode) -> int:
        return 0

    def pb_variable(self, node: ProblemVariableNode) -> int:
        return 1

    def pb_parameter(self, node: ProblemParameterNode) -> int:
        return 0

    def time_shift(self, node: TimeShiftNode) -> int:
        return visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> int:
        return visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> int:
        return visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> int:
        return visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> int:
        scenario_operator_cls = getattr(gems.expression.scenario_operator, node.name)
        # TODO: Carefully check if this formula is correct
        return scenario_operator_cls.degree() * visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> int:
        return 1

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> int:
        return visit(node.operand, self)


def compute_degree(expression: ExpressionNode) -> int:
    return visit(expression, ExpressionDegreeVisitor())


def is_constant(expr: ExpressionNode) -> bool:
    """
    True if the expression has no variable.
    """
    return compute_degree(expr) == 0


def is_linear(expr: ExpressionNode) -> bool:
    """
    True if the expression is linear with respect to variables.
    """
    return compute_degree(expr) <= 1
