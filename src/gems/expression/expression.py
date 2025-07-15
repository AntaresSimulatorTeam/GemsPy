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
Defines the model for generic expressions.
"""
import enum
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import gems.expression.port_operator
import gems.expression.scenario_operator

AnyExpression = Union[int, float, "ExpressionNode"]


@dataclass(frozen=True)
class ExpressionNode:
    """
    Base class for all nodes of the expression AST.

    Operators overloading is provided to help create expressions
    programmatically.

    Examples
        >>> expr = -var('x') + 5 / param('p')
    """

    def __neg__(self) -> "ExpressionNode":
        return NegationNode(self)

    def __add__(self, rhs: Any) -> "ExpressionNode":
        lhs = self
        operands = []
        rhs = _wrap_in_node(rhs)
        operands.extend(lhs.operands if isinstance(lhs, AdditionNode) else [lhs])
        operands.extend(rhs.operands if isinstance(rhs, AdditionNode) else [rhs])
        return AdditionNode(operands)

    def __radd__(self, lhs: Any) -> "ExpressionNode":
        lhs = _wrap_in_node(lhs)
        return lhs + self

    def __sub__(self, rhs: Any) -> "ExpressionNode":
        lhs = self
        operands = []
        rhs = _wrap_in_node(rhs)
        operands.extend(lhs.operands if isinstance(lhs, AdditionNode) else [lhs])
        right_operands = rhs.operands if isinstance(rhs, AdditionNode) else [rhs]
        operands.extend([-o for o in right_operands])
        return AdditionNode(operands)

    def __rsub__(self, lhs: Any) -> "ExpressionNode":
        lhs = _wrap_in_node(lhs)
        return lhs + self

    def __mul__(self, rhs: Any) -> "ExpressionNode":
        return _apply_if_node(rhs, lambda x: MultiplicationNode(self, x))

    def __rmul__(self, lhs: Any) -> "ExpressionNode":
        return _apply_if_node(lhs, lambda x: MultiplicationNode(x, self))

    def __truediv__(self, rhs: Any) -> "ExpressionNode":
        return _apply_if_node(rhs, lambda x: DivisionNode(self, x))

    def __rtruediv__(self, lhs: Any) -> "ExpressionNode":
        return _apply_if_node(lhs, lambda x: DivisionNode(x, self))

    def __le__(self, rhs: Any) -> "ExpressionNode":
        return _apply_if_node(
            rhs, lambda x: ComparisonNode(self, x, Comparator.LESS_THAN)
        )

    def __ge__(self, rhs: Any) -> "ExpressionNode":
        return _apply_if_node(
            rhs, lambda x: ComparisonNode(self, x, Comparator.GREATER_THAN)
        )

    def __eq__(self, rhs: Any) -> "ExpressionNode":  # type: ignore
        return _apply_if_node(rhs, lambda x: ComparisonNode(self, x, Comparator.EQUAL))

    def time_sum(
        self,
        from_shift: Optional[AnyExpression] = None,
        to_shift: Optional[AnyExpression] = None,
    ) -> "ExpressionNode":
        if from_shift is None and to_shift is None:
            return AllTimeSumNode(self)
        if from_shift is None or to_shift is None:
            raise ValueError("Both time bounds of a time sum must be defined.")
        return TimeSumNode(
            operand=self,
            from_time=_wrap_in_node(from_shift),
            to_time=_wrap_in_node(to_shift),
        )

    def sum_connections(self) -> "ExpressionNode":
        if isinstance(self, PortFieldNode):
            return PortFieldAggregatorNode(self, aggregator="PortSum")
        raise ValueError(
            f"sum_connections() applies only for PortFieldNode, whereas the current node is of type {type(self)}."
        )

    def shift(self, shift: AnyExpression) -> "ExpressionNode":
        return TimeShiftNode(self, _wrap_in_node(shift))

    def eval(self, time: AnyExpression) -> "ExpressionNode":
        return TimeEvalNode(self, _wrap_in_node(time))

    def expec(self) -> "ExpressionNode":
        return _apply_if_node(self, lambda x: ScenarioOperatorNode(x, "Expectation"))

    def variance(self) -> "ExpressionNode":
        return _apply_if_node(self, lambda x: ScenarioOperatorNode(x, "Variance"))


def _wrap_in_node(obj: Any) -> ExpressionNode:
    if isinstance(obj, ExpressionNode):
        return obj
    elif isinstance(obj, float) or isinstance(obj, int):
        return LiteralNode(float(obj))
    raise TypeError(f"Unable to wrap {obj} into an expression node")


def _apply_if_node(
    obj: Any, func: Callable[["ExpressionNode"], "ExpressionNode"]
) -> "ExpressionNode":
    if as_node := _wrap_in_node(obj):
        return func(as_node)
    else:
        return NotImplemented


@dataclass(frozen=True, eq=False)
class VariableNode(ExpressionNode):
    name: str


def var(name: str) -> VariableNode:
    return VariableNode(name)


@dataclass(frozen=True, eq=False)
class PortFieldNode(ExpressionNode):
    """
    References a port field.
    """

    port_name: str
    field_name: str


def port_field(port_name: str, field_name: str) -> PortFieldNode:
    return PortFieldNode(port_name, field_name)


@dataclass(frozen=True, eq=False)
class ParameterNode(ExpressionNode):
    name: str


def param(name: str) -> ParameterNode:
    return ParameterNode(name)


@dataclass(frozen=True, eq=False)
class ComponentParameterNode(ExpressionNode):
    """
    Represents one parameter of one component.

    When building actual equations for a system,
    we need to associated each parameter to its
    actual component, at some point.
    """

    component_id: str
    name: str


def comp_param(component_id: str, name: str) -> ComponentParameterNode:
    return ComponentParameterNode(component_id, name)


@dataclass(frozen=True)
class TimeIndex:
    pass


@dataclass(frozen=True)
class NoTimeIndex(TimeIndex):
    """
    Some values do not depend on the timestep, this index should be used for them
    (think one variable for all timesteps).
    """

    pass


@dataclass(frozen=True)
class TimeShift(TimeIndex):
    """
    Represents the current timestep + a shift.

    This should only be used for nodes that actually depend on the timestep,
    never for time-independent nodes (constant parameters ...).
    """

    timeshift: int


@dataclass(frozen=True)
class TimeStep(TimeIndex):
    """
    Represents a given timestep, independently of the current timestep.

    This should only be used for nodes that actually depend on the timestep,
    never for time-independent nodes (constant parameters ...).
    """

    timestep: int


@dataclass(frozen=True)
class ScenarioIndex:
    pass


@dataclass(frozen=True)
class NoScenarioIndex(ScenarioIndex):
    """
    Some values do not depend on the scenario, this index should be used for them
    (think one variable for all timesteps).
    """

    pass


@dataclass(frozen=True)
class CurrentScenarioIndex(ScenarioIndex):
    """
    Represents the current scenario.

    This should only be used for nodes that actually depend on the scenario,
    never for scenario-independent nodes (constant parameters ...).
    """

    pass


@dataclass(frozen=True)
class OneScenarioIndex(ScenarioIndex):
    """
    Represents a given scenario out of all scenarios.

    This should only be used for nodes that actually depend on the scenario,
    never for scenario-independent nodes (constant parameters ...).
    """

    scenario: int


@dataclass(frozen=True, eq=False)
class ProblemParameterNode(ExpressionNode):
    """
    Represents one parameter of the optimization problem
    """

    component_id: str
    name: str
    time_index: TimeIndex
    scenario_index: ScenarioIndex


def problem_param(
    component_id: str, name: str, time_index: TimeIndex, scenario_index: ScenarioIndex
) -> ProblemParameterNode:
    return ProblemParameterNode(component_id, name, time_index, scenario_index)


@dataclass(frozen=True, eq=False)
class ComponentVariableNode(ExpressionNode):
    """
    Represents one variable of one component.

    When building actual equations for a system,
    we need to associated each variable to its
    actual component, at some point.
    """

    component_id: str
    name: str


def comp_var(component_id: str, name: str) -> ComponentVariableNode:
    return ComponentVariableNode(component_id, name)


@dataclass(frozen=True, eq=False)
class ProblemVariableNode(ExpressionNode):
    """
    Represents one variable of the optimization problem
    """

    component_id: str
    name: str
    time_index: TimeIndex
    scenario_index: ScenarioIndex


def problem_var(
    component_id: str, name: str, time_index: TimeIndex, scenario_index: ScenarioIndex
) -> ProblemVariableNode:
    return ProblemVariableNode(component_id, name, time_index, scenario_index)


@dataclass(frozen=True, eq=False)
class LiteralNode(ExpressionNode):
    value: float


def literal(value: float) -> LiteralNode:
    return LiteralNode(value)


def is_unbounded(expr: ExpressionNode) -> bool:
    return isinstance(expr, LiteralNode) and (abs(expr.value) == float("inf"))


def is_non_negative(expr: ExpressionNode) -> bool:
    return isinstance(expr, LiteralNode) and (expr.value >= 0)


@dataclass(frozen=True, eq=False)
class UnaryOperatorNode(ExpressionNode):
    operand: ExpressionNode


@dataclass(frozen=True, eq=False)
class PortFieldAggregatorNode(UnaryOperatorNode):
    aggregator: str

    def __post_init__(self) -> None:
        valid_names = [
            cls.__name__
            for _, cls in inspect.getmembers(
                gems.expression.port_operator, inspect.isclass
            )
            if issubclass(cls, gems.expression.port_operator.PortAggregator)
        ]
        if self.aggregator not in valid_names:
            raise NotImplementedError(
                f"{self.aggregator} is not a valid port aggregator, valid port aggregators are {valid_names}"
            )


@dataclass(frozen=True, eq=False)
class NegationNode(UnaryOperatorNode):
    pass


@dataclass(frozen=True, eq=False)
class BinaryOperatorNode(ExpressionNode):
    left: ExpressionNode
    right: ExpressionNode


class Comparator(enum.Enum):
    LESS_THAN = "LESS_THAN"
    EQUAL = "EQUAL"
    GREATER_THAN = "GREATER_THAN"


@dataclass(frozen=True, eq=False)
class ComparisonNode(BinaryOperatorNode):
    comparator: Comparator


@dataclass(frozen=True, eq=False)
class AdditionNode(ExpressionNode):
    operands: List[ExpressionNode]


@dataclass(frozen=True, eq=False)
class MultiplicationNode(BinaryOperatorNode):
    pass


@dataclass(frozen=True, eq=False)
class DivisionNode(BinaryOperatorNode):
    pass


@dataclass(frozen=True, eq=False)
class TimeShiftNode(UnaryOperatorNode):
    time_shift: ExpressionNode


@dataclass(frozen=True, eq=False)
class TimeEvalNode(UnaryOperatorNode):
    eval_time: ExpressionNode


@dataclass(frozen=True, eq=False)
class TimeSumNode(UnaryOperatorNode):
    from_time: ExpressionNode
    to_time: ExpressionNode


@dataclass(frozen=True, eq=False)
class AllTimeSumNode(UnaryOperatorNode):
    """
    Separate from time sum node because it's actually a quite different operation:
    In particular, this changes the time indexing.
    """

    pass


@dataclass(frozen=True, eq=False)
class ScenarioOperatorNode(UnaryOperatorNode):
    name: str

    def __post_init__(self) -> None:
        valid_names = [
            cls.__name__
            for _, cls in inspect.getmembers(
                gems.expression.scenario_operator, inspect.isclass
            )
            if issubclass(cls, gems.expression.scenario_operator.ScenarioOperator)
        ]
        if self.name not in valid_names:
            raise ValueError(
                f"{self.name} is not a valid scenario operator, valid scenario operators are {valid_names}"
            )


def sum_expressions(expressions: Sequence[ExpressionNode]) -> ExpressionNode:
    if len(expressions) == 0:
        return LiteralNode(0)
    if len(expressions) == 1:
        return expressions[0]
    return AdditionNode([e for e in expressions])
