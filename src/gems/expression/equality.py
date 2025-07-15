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

import math
from dataclasses import dataclass
from typing import Optional

from gems.expression import (
    AdditionNode,
    ComparisonNode,
    DivisionNode,
    ExpressionNode,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    VariableNode,
)
from gems.expression.expression import (
    AllTimeSumNode,
    BinaryOperatorNode,
    ComponentParameterNode,
    ComponentVariableNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
)


@dataclass(frozen=True)
class EqualityVisitor:
    abs_tol: float = 0
    rel_tol: float = 0

    def __post_init__(self) -> None:
        if self.abs_tol < 0:
            raise ValueError(
                f"Absolute comparison tolerance must be >= 0, got {self.abs_tol}"
            )
        if self.rel_tol < 0:
            raise ValueError(
                f"Relative comparison tolerance must be >= 0, got {self.rel_tol}"
            )

    def visit(self, left: ExpressionNode, right: ExpressionNode) -> bool:
        if left.__class__ != right.__class__:
            return False
        if isinstance(left, LiteralNode) and isinstance(right, LiteralNode):
            return self.literal(left, right)
        if isinstance(left, NegationNode) and isinstance(right, NegationNode):
            return self.negation(left, right)
        if isinstance(left, AdditionNode) and isinstance(right, AdditionNode):
            return self.addition(left, right)
        if isinstance(left, DivisionNode) and isinstance(right, DivisionNode):
            return self.division(left, right)
        if isinstance(left, MultiplicationNode) and isinstance(
            right, MultiplicationNode
        ):
            return self.multiplication(left, right)
        if isinstance(left, ComparisonNode) and isinstance(right, ComparisonNode):
            return self.comparison(left, right)
        if isinstance(left, VariableNode) and isinstance(right, VariableNode):
            return self.variable(left, right)
        if isinstance(left, ParameterNode) and isinstance(right, ParameterNode):
            return self.parameter(left, right)
        if isinstance(left, ComponentVariableNode) and isinstance(
            right, ComponentVariableNode
        ):
            return self.comp_variable(left, right)
        if isinstance(left, ComponentParameterNode) and isinstance(
            right, ComponentParameterNode
        ):
            return self.comp_parameter(left, right)
        if isinstance(left, ProblemVariableNode) and isinstance(
            right, ProblemVariableNode
        ):
            return self.problem_variable(left, right)
        if isinstance(left, ProblemParameterNode) and isinstance(
            right, ProblemParameterNode
        ):
            return self.problem_parameter(left, right)
        if isinstance(left, TimeShiftNode) and isinstance(right, TimeShiftNode):
            return self.time_shift(left, right)
        if isinstance(left, TimeEvalNode) and isinstance(right, TimeEvalNode):
            return self.time_eval(left, right)
        if isinstance(left, TimeSumNode) and isinstance(right, TimeSumNode):
            return self.time_sum(left, right)
        if isinstance(left, AllTimeSumNode) and isinstance(right, AllTimeSumNode):
            return self.all_time_sum(left, right)
        if isinstance(left, ScenarioOperatorNode) and isinstance(
            right, ScenarioOperatorNode
        ):
            return self.scenario_operator(left, right)
        if isinstance(left, PortFieldNode) and isinstance(right, PortFieldNode):
            return self.port_field(left, right)
        if isinstance(left, PortFieldAggregatorNode) and isinstance(
            right, PortFieldAggregatorNode
        ):
            return self.port_field_aggregator(left, right)
        raise NotImplementedError(f"Equality not implemented for {left.__class__}")

    def literal(self, left: LiteralNode, right: LiteralNode) -> bool:
        return math.isclose(
            left.value, right.value, abs_tol=self.abs_tol, rel_tol=self.rel_tol
        )

    def _visit_operands(
        self, left: BinaryOperatorNode, right: BinaryOperatorNode
    ) -> bool:
        return self.visit(left.left, right.left) and self.visit(left.right, right.right)

    def negation(self, left: NegationNode, right: NegationNode) -> bool:
        return self.visit(left.operand, right.operand)

    def addition(self, left: AdditionNode, right: AdditionNode) -> bool:
        left_ops = left.operands
        right_ops = right.operands
        return len(left_ops) == len(right_ops) and all(
            self.visit(l, r) for l, r in zip(left_ops, right_ops)
        )

    def multiplication(
        self, left: MultiplicationNode, right: MultiplicationNode
    ) -> bool:
        return self._visit_operands(left, right)

    def division(self, left: DivisionNode, right: DivisionNode) -> bool:
        return self._visit_operands(left, right)

    def comparison(self, left: ComparisonNode, right: ComparisonNode) -> bool:
        return left.comparator == right.comparator and self._visit_operands(left, right)

    def variable(self, left: VariableNode, right: VariableNode) -> bool:
        return left.name == right.name

    def parameter(self, left: ParameterNode, right: ParameterNode) -> bool:
        return left.name == right.name

    def comp_variable(
        self, left: ComponentVariableNode, right: ComponentVariableNode
    ) -> bool:
        return left.name == right.name and left.component_id == right.component_id

    def comp_parameter(
        self, left: ComponentParameterNode, right: ComponentParameterNode
    ) -> bool:
        return left.name == right.name and left.component_id == right.component_id

    def problem_variable(
        self, left: ProblemVariableNode, right: ProblemVariableNode
    ) -> bool:
        return (
            left.name == right.name
            and left.component_id == right.component_id
            and left.time_index == right.time_index
            and left.scenario_index == right.scenario_index
        )

    def problem_parameter(
        self, left: ProblemParameterNode, right: ProblemParameterNode
    ) -> bool:
        return (
            left.name == right.name
            and left.component_id == right.component_id
            and left.time_index == right.time_index
            and left.scenario_index == right.scenario_index
        )

    def time_shift(self, left: TimeShiftNode, right: TimeShiftNode) -> bool:
        return self.visit(left.time_shift, right.time_shift) and self.visit(
            left.operand, right.operand
        )

    def time_eval(self, left: TimeEvalNode, right: TimeEvalNode) -> bool:
        return self.visit(left.eval_time, right.eval_time) and self.visit(
            left.operand, right.operand
        )

    def time_sum(self, left: TimeSumNode, right: TimeSumNode) -> bool:
        return (
            self.visit(left.from_time, right.from_time)
            and self.visit(left.to_time, right.to_time)
            and self.visit(left.operand, right.operand)
        )

    def all_time_sum(self, left: AllTimeSumNode, right: AllTimeSumNode) -> bool:
        return self.visit(left.operand, right.operand)

    def scenario_operator(
        self, left: ScenarioOperatorNode, right: ScenarioOperatorNode
    ) -> bool:
        return left.name == right.name and self.visit(left.operand, right.operand)

    def port_field(self, left: PortFieldNode, right: PortFieldNode) -> bool:
        return left.port_name == right.port_name and left.field_name == right.field_name

    def port_field_aggregator(
        self, left: PortFieldAggregatorNode, right: PortFieldAggregatorNode
    ) -> bool:
        return left.aggregator == right.aggregator and self.visit(
            left.operand, right.operand
        )


def expressions_equal(
    left: ExpressionNode, right: ExpressionNode, abs_tol: float = 0, rel_tol: float = 0
) -> bool:
    """
    True if both expression nodes are equal. Literal values may be compared with absolute or relative tolerance.
    """
    return EqualityVisitor(abs_tol, rel_tol).visit(left, right)


def expressions_equal_if_present(
    lhs: Optional[ExpressionNode], rhs: Optional[ExpressionNode]
) -> bool:
    if lhs is None and rhs is None:
        return True
    elif lhs is None or rhs is None:
        return False
    else:
        return expressions_equal(lhs, rhs)
