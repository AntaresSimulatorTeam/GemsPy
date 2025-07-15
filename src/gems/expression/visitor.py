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
Defines abstract base class for visitors of expressions.
"""
import typing
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from gems.expression.expression import (
    AdditionNode,
    AllTimeSumNode,
    ComparisonNode,
    ComponentParameterNode,
    ComponentVariableNode,
    DivisionNode,
    ExpressionNode,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
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

T = TypeVar("T")


class ExpressionVisitor(ABC, Generic[T]):
    """
    Abstract base class for visitors of expressions.

    Visitor can be implemented to carry out arbitrary kind
    of analysis over an expression, such as evaluation,
    manipulation ...
    """

    @abstractmethod
    def literal(self, node: LiteralNode) -> T:
        ...

    @abstractmethod
    def negation(self, node: NegationNode) -> T:
        ...

    @abstractmethod
    def addition(self, node: AdditionNode) -> T:
        ...

    @abstractmethod
    def multiplication(self, node: MultiplicationNode) -> T:
        ...

    @abstractmethod
    def division(self, node: DivisionNode) -> T:
        ...

    @abstractmethod
    def comparison(self, node: ComparisonNode) -> T:
        ...

    @abstractmethod
    def variable(self, node: VariableNode) -> T:
        ...

    @abstractmethod
    def parameter(self, node: ParameterNode) -> T:
        ...

    @abstractmethod
    def comp_parameter(self, node: ComponentParameterNode) -> T:
        ...

    @abstractmethod
    def comp_variable(self, node: ComponentVariableNode) -> T:
        ...

    @abstractmethod
    def pb_parameter(self, node: ProblemParameterNode) -> T:
        ...

    @abstractmethod
    def pb_variable(self, node: ProblemVariableNode) -> T:
        ...

    @abstractmethod
    def time_shift(self, node: TimeShiftNode) -> T:
        ...

    @abstractmethod
    def time_eval(self, node: TimeEvalNode) -> T:
        ...

    @abstractmethod
    def time_sum(self, node: TimeSumNode) -> T:
        ...

    @abstractmethod
    def all_time_sum(self, node: AllTimeSumNode) -> T:
        ...

    @abstractmethod
    def scenario_operator(self, node: ScenarioOperatorNode) -> T:
        ...

    @abstractmethod
    def port_field(self, node: PortFieldNode) -> T:
        ...

    @abstractmethod
    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> T:
        ...


def visit(root: ExpressionNode, visitor: ExpressionVisitor[T]) -> T:
    """
    Utility method to dispatch calls to the right method of a visitor.
    """
    if isinstance(root, LiteralNode):
        return visitor.literal(root)
    elif isinstance(root, NegationNode):
        return visitor.negation(root)
    elif isinstance(root, VariableNode):
        return visitor.variable(root)
    elif isinstance(root, ParameterNode):
        return visitor.parameter(root)
    elif isinstance(root, ComponentParameterNode):
        return visitor.comp_parameter(root)
    elif isinstance(root, ComponentVariableNode):
        return visitor.comp_variable(root)
    elif isinstance(root, ProblemParameterNode):
        return visitor.pb_parameter(root)
    elif isinstance(root, ProblemVariableNode):
        return visitor.pb_variable(root)
    elif isinstance(root, AdditionNode):
        return visitor.addition(root)
    elif isinstance(root, MultiplicationNode):
        return visitor.multiplication(root)
    elif isinstance(root, DivisionNode):
        return visitor.division(root)
    elif isinstance(root, ComparisonNode):
        return visitor.comparison(root)
    elif isinstance(root, TimeShiftNode):
        return visitor.time_shift(root)
    elif isinstance(root, TimeEvalNode):
        return visitor.time_eval(root)
    elif isinstance(root, TimeSumNode):
        return visitor.time_sum(root)
    elif isinstance(root, AllTimeSumNode):
        return visitor.all_time_sum(root)
    elif isinstance(root, ScenarioOperatorNode):
        return visitor.scenario_operator(root)
    elif isinstance(root, PortFieldNode):
        return visitor.port_field(root)
    elif isinstance(root, PortFieldAggregatorNode):
        return visitor.port_field_aggregator(root)
    raise ValueError(f"Unknown expression node type {root.__class__}")


class SupportsOperations(Protocol[T]):
    """
    Defines a type which implements math operations +, -, *, /
    """

    @abstractmethod
    def __add__(self, other: T) -> T:
        pass

    @abstractmethod
    def __neg__(self) -> T:
        pass

    @abstractmethod
    def __sub__(self, other: T) -> T:
        pass

    @abstractmethod
    def __mul__(self, other: T) -> T:
        pass

    @abstractmethod
    def __truediv__(self, other: T) -> T:
        pass


T_op = TypeVar("T_op", bound=SupportsOperations)


class ExpressionVisitorOperations(ExpressionVisitor[T_op], ABC):
    """
    Provides default implementations of math operations
    based on (+, -, /, *) operations of type T.
    """

    def negation(self, node: NegationNode) -> T_op:
        return -visit(node.operand, self)

    def addition(self, node: AdditionNode) -> T_op:
        operands = [visit(o, self) for o in node.operands]
        res = operands[0]
        for o in operands[1:]:
            res = res + o
        return res

    def multiplication(self, node: MultiplicationNode) -> T_op:
        left_value = visit(node.left, self)
        right_value = visit(node.right, self)
        return left_value * right_value

    def division(self, node: DivisionNode) -> T_op:
        left_value = visit(node.left, self)
        right_value = visit(node.right, self)
        return left_value / right_value
