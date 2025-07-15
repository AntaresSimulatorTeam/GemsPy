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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

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
    ComparisonNode,
    ExpressionNode,
    LiteralNode,
    ParameterNode,
    ScenarioOperatorNode,
    VariableNode,
)
from .indexing import IndexingStructureProvider
from .visitor import ExpressionVisitorOperations, visit


class ValueProvider(ABC):
    """
    Implementations are in charge of mapping parameters and variables to their values.
    Depending on the implementation, evaluation may require a component id or not.
    """

    @abstractmethod
    def get_variable_value(self, name: str) -> float:
        ...

    @abstractmethod
    def get_parameter_value(self, name: str) -> float:
        ...

    @abstractmethod
    def get_component_variable_value(self, component_id: str, name: str) -> float:
        ...

    @abstractmethod
    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        ...


@dataclass(frozen=True)
class EvaluationContext(ValueProvider):
    """
    Simple value provider relying on dictionaries.
    Does not support component variables/parameters.
    """

    variables: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)

    def get_variable_value(self, name: str) -> float:
        return self.variables[name]

    def get_parameter_value(self, name: str) -> float:
        return self.parameters[name]

    def get_component_variable_value(self, component_id: str, name: str) -> float:
        raise NotImplementedError()

    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        raise NotImplementedError()


@dataclass(frozen=True)
class EvaluationVisitor(ExpressionVisitorOperations[float]):
    """
    Evaluates the expression with respect to the provided context
    (variables and parameters values).
    """

    context: ValueProvider

    def literal(self, node: LiteralNode) -> float:
        return node.value

    def comparison(self, node: ComparisonNode) -> float:
        raise ValueError("Cannot evaluate comparison operator.")

    def variable(self, node: VariableNode) -> float:
        return self.context.get_variable_value(node.name)

    def parameter(self, node: ParameterNode) -> float:
        return self.context.get_parameter_value(node.name)

    def comp_parameter(self, node: ComponentParameterNode) -> float:
        return self.context.get_component_parameter_value(node.component_id, node.name)

    def comp_variable(self, node: ComponentVariableNode) -> float:
        return self.context.get_component_variable_value(node.component_id, node.name)

    def pb_parameter(self, node: ProblemParameterNode) -> float:
        raise ValueError("Should not reach here.")

    def pb_variable(self, node: ProblemVariableNode) -> float:
        raise ValueError("Should not reach here.")

    def time_shift(self, node: TimeShiftNode) -> float:
        raise NotImplementedError()

    def time_eval(self, node: TimeEvalNode) -> float:
        raise NotImplementedError()

    def time_sum(self, node: TimeSumNode) -> float:
        raise NotImplementedError()

    def all_time_sum(self, node: AllTimeSumNode) -> float:
        raise NotImplementedError()

    def scenario_operator(self, node: ScenarioOperatorNode) -> float:
        raise NotImplementedError()

    def port_field(self, node: PortFieldNode) -> float:
        raise NotImplementedError()

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> float:
        raise NotImplementedError()


def evaluate(expression: ExpressionNode, value_provider: ValueProvider) -> float:
    return visit(expression, EvaluationVisitor(value_provider))
