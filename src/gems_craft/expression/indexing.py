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

from gems_craft.expression.indexing_structure import IndexingStructure

from .expression import (
    AbsNode,
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    DivisionNode,
    DualNode,
    ExpressionNode,
    FloorNode,
    LiteralNode,
    LowerBoundNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ReducedCostNode,
    RoundNode,
    ScenarioOperatorNode,
    SetIndexNode,
    SumOverNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    UpperBoundNode,
    VariableNode,
)
from .visitor import ExpressionVisitor, T, visit


class IndexingStructureProvider(ABC):
    @abstractmethod
    def get_parameter_structure(self, name: str) -> IndexingStructure: ...

    @abstractmethod
    def get_variable_structure(self, name: str) -> IndexingStructure: ...

    @abstractmethod
    def get_constraint_structure(self, name: str) -> IndexingStructure: ...


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
        res = IndexingStructure(False, False)
        for o in operands:
            res = res | visit(o, self)
        return res

    def addition(self, node: AdditionNode) -> IndexingStructure:
        return self._combine(node.operands)

    def multiplication(self, node: MultiplicationNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def division(self, node: DivisionNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def comparison(self, node: ComparisonNode) -> IndexingStructure:
        return self._combine([node.left, node.right])

    def variable(self, node: VariableNode) -> IndexingStructure:
        return self.context.get_variable_structure(node.name)

    def parameter(self, node: ParameterNode) -> IndexingStructure:
        return self.context.get_parameter_structure(node.name)

    def time_shift(self, node: TimeShiftNode) -> IndexingStructure:
        return visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> IndexingStructure:
        inner = visit(node.operand, self)
        return IndexingStructure(False, inner.scenario, inner.sets)

    def time_sum(self, node: TimeSumNode) -> IndexingStructure:
        return visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> IndexingStructure:
        inner = visit(node.operand, self)
        return IndexingStructure(False, inner.scenario, inner.sets)

    def set_index(self, node: SetIndexNode) -> IndexingStructure:
        inner = visit(node.operand, self)
        if node.position is not None:
            # Explicit position resolves to a single element: collapses the
            # dimension, mirroring how time_eval collapses time.
            return IndexingStructure(inner.time, inner.scenario, inner.sets - {node.set_id})
        # Bare (`X[fuel]`) or relative-shift (`X[fuel+1]`) forms still vary
        # over the set, mirroring how time_shift keeps time.
        return inner

    def sum_over(self, node: SumOverNode) -> IndexingStructure:
        inner = visit(node.operand, self)
        return IndexingStructure(inner.time, inner.scenario, inner.sets - {node.set_id})

    def scenario_operator(self, node: ScenarioOperatorNode) -> IndexingStructure:
        inner = visit(node.operand, self)
        return IndexingStructure(inner.time, False, inner.sets)

    def port_field(self, node: PortFieldNode) -> IndexingStructure:
        raise ValueError(
            "Port fields must be resolved before computing indexing structure."
        )

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> IndexingStructure:
        raise ValueError(
            "Port fields aggregators must be resolved before computing indexing structure."
        )

    def floor(self, node: FloorNode) -> IndexingStructure:
        return visit(node.operand, self)

    def ceil(self, node: CeilNode) -> IndexingStructure:
        return visit(node.operand, self)

    def abs(self, node: AbsNode) -> IndexingStructure:
        return visit(node.operand, self)

    def round(self, node: RoundNode) -> IndexingStructure:
        return visit(node.operand, self)

    def maximum(self, node: MaxNode) -> IndexingStructure:
        return self._combine(node.operands)

    def minimum(self, node: MinNode) -> IndexingStructure:
        return self._combine(node.operands)

    def dual(self, node: DualNode) -> IndexingStructure:
        return self.context.get_constraint_structure(node.constraint_id)

    def reduced_cost(self, node: ReducedCostNode) -> IndexingStructure:
        return self.context.get_variable_structure(node.variable_id)

    def lower_bound(self, node: LowerBoundNode) -> IndexingStructure:
        return self.context.get_variable_structure(node.variable_id)

    def upper_bound(self, node: UpperBoundNode) -> IndexingStructure:
        return self.context.get_variable_structure(node.variable_id)


def compute_indexation(
    expression: ExpressionNode, provider: IndexingStructureProvider
) -> IndexingStructure:
    return visit(expression, TimeScenarioIndexingVisitor(provider))
