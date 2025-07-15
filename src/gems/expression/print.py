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
from typing import Dict

from gems.expression.expression import (
    AllTimeSumNode,
    ComponentParameterNode,
    ComponentVariableNode,
    ExpressionNode,
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
    Comparator,
    ComparisonNode,
    DivisionNode,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    ScenarioOperatorNode,
    VariableNode,
)
from .visitor import ExpressionVisitor, visit

_COMPARISON_OPERATOR_TO_STRING: Dict[Comparator, str] = {
    Comparator.LESS_THAN: "<=",
    Comparator.EQUAL: "==",
    Comparator.GREATER_THAN: ">=",
}


@dataclass(frozen=True)
class PrinterVisitor(ExpressionVisitor[str]):
    """
    Produces a string representing the mathematical expression.

    TODO: remove parenthis where not necessary.
    """

    def literal(self, node: LiteralNode) -> str:
        return str(node.value)

    def negation(self, node: NegationNode) -> str:
        return f"-({visit(node.operand, self)})"

    def addition(self, node: AdditionNode) -> str:
        if len(node.operands) == 0:
            return ""
        res = visit(node.operands[0], self)
        for o in node.operands[1:]:
            if isinstance(o, NegationNode):
                res += f" - {visit(o.operand, self)}"
            else:
                res += f" + {visit(o, self)}"
        return f"({res})"

    def multiplication(self, node: MultiplicationNode) -> str:
        left_value = visit(node.left, self)
        right_value = visit(node.right, self)
        return f"({left_value} * {right_value})"

    def division(self, node: DivisionNode) -> str:
        left_value = visit(node.left, self)
        right_value = visit(node.right, self)
        return f"({left_value} / {right_value})"

    def comparison(self, node: ComparisonNode) -> str:
        op = _COMPARISON_OPERATOR_TO_STRING[node.comparator]
        left_value = visit(node.left, self)
        right_value = visit(node.right, self)
        return f"{left_value} {op} {right_value}"

    def variable(self, node: VariableNode) -> str:
        return node.name

    def parameter(self, node: ParameterNode) -> str:
        return node.name

    def comp_variable(self, node: ComponentVariableNode) -> str:
        return f"{node.component_id}.{node.name}"

    def comp_parameter(self, node: ComponentParameterNode) -> str:
        return f"{node.component_id}.{node.name}"

    def pb_variable(self, node: ProblemVariableNode) -> str:
        # TODO
        return f"{node.component_id}.{node.name}"

    def pb_parameter(self, node: ProblemParameterNode) -> str:
        # TODO
        return f"{node.component_id}.{node.name}"

    def time_shift(self, node: TimeShiftNode) -> str:
        return f"({visit(node.operand, self)}.shift({visit(node.time_shift, self)}))"

    def time_eval(self, node: TimeEvalNode) -> str:
        return f"({visit(node.operand, self)}.eval({visit(node.eval_time, self)}))"

    def time_sum(self, node: TimeSumNode) -> str:
        return f"({visit(node.operand, self)}.time_sum({visit(node.from_time, self)}, {visit(node.to_time, self)}))"

    def all_time_sum(self, node: AllTimeSumNode) -> str:
        return f"({visit(node.operand, self)}.time_sum())"

    def scenario_operator(self, node: ScenarioOperatorNode) -> str:
        return f"({visit(node.operand, self)}.{str(node.name)})"

    def port_field(self, node: PortFieldNode) -> str:
        return f"{node.port_name}.{node.field_name}"

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> str:
        return f"({visit(node.operand, self)}.{node.aggregator})"


def print_expr(expression: ExpressionNode) -> str:
    return visit(expression, PrinterVisitor())
