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

from dataclasses import dataclass
from typing import List, cast

from .expression import (
    AllTimeSumNode,
    ComparisonNode,
    ComponentParameterNode,
    ComponentVariableNode,
    ExpressionNode,
    LiteralNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    VariableNode,
)
from .visitor import ExpressionVisitorOperations, visit


@dataclass(frozen=True)
class CopyVisitor(ExpressionVisitorOperations[ExpressionNode]):
    """
    Simply copies the whole AST.
    """

    def literal(self, node: LiteralNode) -> ExpressionNode:
        return LiteralNode(node.value)

    def comparison(self, node: ComparisonNode) -> ExpressionNode:
        return ComparisonNode(
            visit(node.left, self), visit(node.right, self), node.comparator
        )

    def variable(self, node: VariableNode) -> ExpressionNode:
        return VariableNode(node.name)

    def parameter(self, node: ParameterNode) -> ExpressionNode:
        return ParameterNode(node.name)

    def comp_variable(self, node: ComponentVariableNode) -> ExpressionNode:
        return ComponentVariableNode(node.component_id, node.name)

    def comp_parameter(self, node: ComponentParameterNode) -> ExpressionNode:
        return ComponentParameterNode(node.component_id, node.name)

    def pb_variable(self, node: ProblemVariableNode) -> ExpressionNode:
        return ProblemVariableNode(
            node.component_id, node.name, node.time_index, node.scenario_index
        )

    def pb_parameter(self, node: ProblemParameterNode) -> ExpressionNode:
        return ProblemParameterNode(
            node.component_id, node.name, node.time_index, node.scenario_index
        )

    def time_shift(self, node: TimeShiftNode) -> ExpressionNode:
        return TimeShiftNode(visit(node.operand, self), visit(node.time_shift, self))

    def time_eval(self, node: TimeEvalNode) -> ExpressionNode:
        return TimeEvalNode(visit(node.operand, self), visit(node.eval_time, self))

    def time_sum(self, node: TimeSumNode) -> ExpressionNode:
        return TimeSumNode(
            visit(node.operand, self),
            visit(node.from_time, self),
            visit(node.to_time, self),
        )

    def all_time_sum(self, node: AllTimeSumNode) -> ExpressionNode:
        return AllTimeSumNode(visit(node.operand, self))

    def scenario_operator(self, node: ScenarioOperatorNode) -> ExpressionNode:
        return ScenarioOperatorNode(visit(node.operand, self), node.name)

    def port_field(self, node: PortFieldNode) -> ExpressionNode:
        return PortFieldNode(node.port_name, node.field_name)

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> ExpressionNode:
        return PortFieldAggregatorNode(visit(node.operand, self), node.aggregator)


def copy_expression(expression: ExpressionNode) -> ExpressionNode:
    return visit(expression, CopyVisitor())
