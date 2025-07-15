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
from dataclasses import dataclass
from typing import List

from gems.expression.indexing_structure import IndexingStructure

from .expression import (
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
from .visitor import ExpressionVisitor, T, visit


class IndexingStructureProvider(ABC):
    @abstractmethod
    def get_parameter_structure(self, name: str) -> IndexingStructure:
        ...

    @abstractmethod
    def get_variable_structure(self, name: str) -> IndexingStructure:
        ...

    @abstractmethod
    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        ...

    @abstractmethod
    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        ...


@dataclass(frozen=True)
class TimeScenarioIndexingVisitor(ExpressionVisitor[IndexingStructure]):
    """
    Determines if the expression represents a single expression or an expression that should be instantiated for all time steps.
    """

    context: IndexingStructureProvider

    def literal(self, node: LiteralNode) -> IndexingStructure:
        return IndexingStructure(False, False)

    def negation(self, node: NegationNode) -> IndexingStructure:
        return visit(node.operand, self)

    def _combine(self, operands: List[ExpressionNode]) -> IndexingStructure:
        if not operands:
            return IndexingStructure(False, False)
        res = visit(operands[0], self)
        if res.is_time_scenario_varying():
            return res
        for o in operands[1:]:
            res = res | visit(o, self)
            if res.is_time_scenario_varying():
                return res
        return res

    def addition(self, node: AdditionNode) -> IndexingStructure:
        # performance note:
        # here we don't need to visit all nodes, we can stop as soon as
        # index is true/true
        return self._combine(node.operands)

    def multiplication(self, node: MultiplicationNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def division(self, node: DivisionNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def comparison(self, node: ComparisonNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def variable(self, node: VariableNode) -> IndexingStructure:
        time = self.context.get_variable_structure(node.name).time == True
        scenario = self.context.get_variable_structure(node.name).scenario == True
        return IndexingStructure(time, scenario)

    def parameter(self, node: ParameterNode) -> IndexingStructure:
        time = self.context.get_parameter_structure(node.name).time == True
        scenario = self.context.get_parameter_structure(node.name).scenario == True
        return IndexingStructure(time, scenario)

    def comp_variable(self, node: ComponentVariableNode) -> IndexingStructure:
        return self.context.get_component_variable_structure(
            node.component_id, node.name
        )

    def comp_parameter(self, node: ComponentParameterNode) -> IndexingStructure:
        return self.context.get_component_parameter_structure(
            node.component_id, node.name
        )

    def pb_variable(self, node: ProblemVariableNode) -> IndexingStructure:
        raise ValueError(
            "Not relevant to compute indexation on already instantiated problem variables."
        )

    def pb_parameter(self, node: ProblemParameterNode) -> IndexingStructure:
        raise ValueError(
            "Not relevant to compute indexation on already instantiated problem parameters."
        )

    def time_shift(self, node: TimeShiftNode) -> IndexingStructure:
        return visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> IndexingStructure:
        return IndexingStructure(False, visit(node.operand, self).scenario)

    def time_sum(self, node: TimeSumNode) -> IndexingStructure:
        return visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> IndexingStructure:
        return IndexingStructure(False, visit(node.operand, self).scenario)

    def scenario_operator(self, node: ScenarioOperatorNode) -> IndexingStructure:
        return IndexingStructure(visit(node.operand, self).time, False)

    def port_field(self, node: PortFieldNode) -> IndexingStructure:
        raise ValueError(
            "Port fields must be resolved before computing indexing structure."
        )

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> IndexingStructure:
        raise ValueError(
            "Port fields aggregators must be resolved before computing indexing structure."
        )


def compute_indexation(
    expression: ExpressionNode, provider: IndexingStructureProvider
) -> IndexingStructure:
    return visit(expression, TimeScenarioIndexingVisitor(provider))
