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
from typing import Any, Dict, List, Optional, Union

from gems.expression import (
    AdditionNode,
    DivisionNode,
    ExpressionVisitor,
    MultiplicationNode,
    NegationNode,
)
from gems.expression.expression import (
    AllTimeSumNode,
    ComparisonNode,
    ComponentParameterNode,
    ComponentVariableNode,
    CurrentScenarioIndex,
    ExpressionNode,
    LiteralNode,
    NoScenarioIndex,
    NoTimeIndex,
    OneScenarioIndex,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    ScenarioIndex,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeIndex,
    TimeShift,
    TimeShiftNode,
    TimeStep,
    TimeSumNode,
    VariableNode,
)
from gems.expression.visitor import visit
from gems.simulation.linear_expression import LinearExpression, Term, TermKey


class ParameterGetter(ABC):
    @abstractmethod
    def get_parameter_value(
        self,
        component_id: str,
        parameter_name: str,
        timestep: Optional[int],
        scenario: Optional[int],
    ) -> float:
        pass


@dataclass
class MutableTerm:
    coefficient: float
    component_id: str
    variable_name: str
    time_index: Optional[int]
    scenario_index: Optional[int]

    def to_key(self) -> TermKey:
        return TermKey(
            self.component_id,
            self.variable_name,
            self.time_index,
            self.scenario_index,
        )

    def to_term(self) -> Term:
        return Term(
            self.coefficient,
            self.component_id,
            self.variable_name,
            self.time_index,
            self.scenario_index,
        )


@dataclass
class LinearExpressionData:
    terms: List[MutableTerm]
    constant: float

    def build(self) -> LinearExpression:
        res_terms: Dict[TermKey, Any] = {}
        for t in self.terms:
            k = t.to_key()
            if k in res_terms:
                current_t = res_terms[k]
                current_t.coefficient += t.coefficient
            else:
                res_terms[k] = t
        for k, v in res_terms.items():
            res_terms[k] = v.to_term()
        return LinearExpression(res_terms, self.constant)


@dataclass(frozen=True)
class LinearExpressionBuilder(ExpressionVisitor[LinearExpressionData]):
    """
    Reduces a generic expression to a linear expression.

    The input expression must respect the constraints of the output of
    the operators expansion expression:
    it must only contain `ProblemVariableNode` for variables
    and `ProblemParameterNode` parameters. It cannot contain anymore
    time aggregators or scenario aggregators, nor port-related nodes.
    """

    # TODO: linear expressions should be re-usable for different timesteps and scenarios
    timestep: Optional[int]
    scenario: Optional[int]
    value_provider: Optional[ParameterGetter] = None

    def negation(self, node: NegationNode) -> LinearExpressionData:
        operand = visit(node.operand, self)
        operand.constant = -operand.constant
        for t in operand.terms:
            t.coefficient = -t.coefficient
        return operand

    def addition(self, node: AdditionNode) -> LinearExpressionData:
        operands = [visit(o, self) for o in node.operands]
        terms = []
        constant: float = 0
        for o in operands:
            constant += o.constant
            terms.extend(o.terms)
        return LinearExpressionData(terms=terms, constant=constant)

    def multiplication(self, node: MultiplicationNode) -> LinearExpressionData:
        lhs = visit(node.left, self)
        rhs = visit(node.right, self)
        if not lhs.terms:
            multiplier = lhs.constant
            actual_expr = rhs
        elif not rhs.terms:
            multiplier = rhs.constant
            actual_expr = lhs
        else:
            raise ValueError(
                "At least one operand of a multiplication must be a constant expression."
            )
        actual_expr.constant *= multiplier
        for t in actual_expr.terms:
            t.coefficient *= multiplier
        return actual_expr

    def division(self, node: DivisionNode) -> LinearExpressionData:
        lhs = visit(node.left, self)
        rhs = visit(node.right, self)
        if rhs.terms:
            raise ValueError(
                "The second operand of a division must be a constant expression."
            )
        divider = rhs.constant
        actual_expr = lhs
        actual_expr.constant /= divider
        for t in actual_expr.terms:
            t.coefficient /= divider
        return actual_expr

    def _get_timestep(self, time_index: TimeIndex) -> Optional[int]:
        if isinstance(time_index, TimeShift):
            if self.timestep is None:
                raise ValueError("Cannot shift a time-independent expression.")
            return self.timestep + time_index.timeshift
        if isinstance(time_index, TimeStep):
            return time_index.timestep
        if isinstance(time_index, NoTimeIndex):
            return None
        else:
            raise TypeError(f"Type {type(time_index)} is not a valid TimeIndex type.")

    def _get_scenario(self, scenario_index: ScenarioIndex) -> Optional[int]:
        if isinstance(scenario_index, OneScenarioIndex):
            return scenario_index.scenario
        elif isinstance(scenario_index, CurrentScenarioIndex):
            return self.scenario
        elif isinstance(scenario_index, NoScenarioIndex):
            return None
        else:
            raise TypeError(
                f"Type {type(scenario_index)} is not a valid ScenarioIndex type."
            )

    def literal(self, node: LiteralNode) -> LinearExpressionData:
        return LinearExpressionData([], node.value)

    def comparison(self, node: ComparisonNode) -> LinearExpressionData:
        raise ValueError("Linear expression cannot contain a comparison operator.")

    def variable(self, node: VariableNode) -> LinearExpressionData:
        raise ValueError(
            "Variables need to be associated with their component ID before linearization."
        )

    def parameter(self, node: ParameterNode) -> LinearExpressionData:
        raise ValueError("Parameters must be evaluated before linearization.")

    def comp_variable(self, node: ComponentVariableNode) -> LinearExpressionData:
        raise ValueError(
            "Variables need to be associated with their timestep/scenario before linearization."
        )

    def pb_variable(self, node: ProblemVariableNode) -> LinearExpressionData:
        return LinearExpressionData(
            [
                MutableTerm(
                    1,
                    node.component_id,
                    node.name,
                    time_index=self._get_timestep(node.time_index),
                    scenario_index=self._get_scenario(node.scenario_index),
                )
            ],
            0,
        )

    def comp_parameter(self, node: ComponentParameterNode) -> LinearExpressionData:
        raise ValueError(
            "Parameters need to be associated with their timestep/scenario before linearization."
        )

    def pb_parameter(self, node: ProblemParameterNode) -> LinearExpressionData:
        # TODO SL: not the best place to do this.
        # in the future, we should evaluate coefficients of variables as time vectors once for all timesteps
        time_index = self._get_timestep(node.time_index)
        scenario_index = self._get_scenario(node.scenario_index)
        return LinearExpressionData(
            [],
            self._value_provider().get_parameter_value(
                node.component_id, node.name, time_index, scenario_index
            ),
        )

    def time_eval(self, node: TimeEvalNode) -> LinearExpressionData:
        raise ValueError("Time operators need to be expanded before linearization.")

    def time_shift(self, node: TimeShiftNode) -> LinearExpressionData:
        raise ValueError("Time operators need to be expanded before linearization.")

    def time_sum(self, node: TimeSumNode) -> LinearExpressionData:
        raise ValueError("Time operators need to be expanded before linearization.")

    def all_time_sum(self, node: AllTimeSumNode) -> LinearExpressionData:
        raise ValueError("Time operators need to be expanded before linearization.")

    def _value_provider(self) -> ParameterGetter:
        if self.value_provider is None:
            raise ValueError(
                "A value provider must be specified to linearize a time operator node."
                " This is required in order to evaluate the value of potential parameters"
                " used to specified the time ids on which the time operator applies."
            )
        return self.value_provider

    def scenario_operator(self, node: ScenarioOperatorNode) -> LinearExpressionData:
        raise ValueError("Scenario operators need to be expanded before linearization.")

    def port_field(self, node: PortFieldNode) -> LinearExpressionData:
        raise ValueError("Port fields must be replaced before linearization.")

    def port_field_aggregator(
        self, node: PortFieldAggregatorNode
    ) -> LinearExpressionData:
        raise ValueError(
            "Port fields aggregators must be replaced before linearization."
        )


def linearize_expression(
    expression: ExpressionNode,
    timestep: Optional[int],
    scenario: Optional[int],
    value_provider: Optional[ParameterGetter] = None,
) -> LinearExpression:
    return visit(
        expression,
        LinearExpressionBuilder(
            value_provider=value_provider, timestep=timestep, scenario=scenario
        ),
    ).build()
