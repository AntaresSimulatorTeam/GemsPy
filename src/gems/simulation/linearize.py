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
from typing import Any, Dict, Iterable, List, Optional

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
class LinearExpressionBuilder(ExpressionVisitor[List[List[LinearExpressionData]]]):
    """
    Reduces a generic expression to a linear expression.

    The input expression must respect the constraints of the output of
    the operators expansion expression:
    it must only contain `ProblemVariableNode` for variables
    and `ProblemParameterNode` parameters. It cannot contain anymore
    time aggregators or scenario aggregators, nor port-related nodes.

    Returns a list of lists of LinearExpressionData, one for each timestep and scenario: access to timestep t, scenario w with linear_expr_data[t][w]
    Use list as access may be faster than map, but much less readable code...
    """

    timesteps: Optional[Iterable[int]] = None
    scenarios: Optional[Iterable[int]] = None
    value_provider: Optional[ParameterGetter] = None

    ### There still many copies of LinearExpressionData objects that may be avoided by rather updating existing ones rather than creating new ones
    def negation(self, node: NegationNode) -> List[List[LinearExpressionData]]:
        operand_all_time_scenario = visit(node.operand, self)
        for operand_given_time in operand_all_time_scenario:
            for op in operand_given_time:
                op.constant = -op.constant
                for t in op.terms:
                    t.coefficient = -t.coefficient
        return operand_all_time_scenario

    def addition(self, node: AdditionNode) -> List[List[LinearExpressionData]]:
        operands_linear_expr_data = [visit(o, self) for o in node.operands]
        linear_expr_datas = []
        for timestep in range(len(operands_linear_expr_data[0])):
            linear_expr_data_given_time = []
            for scenario in range(len(operands_linear_expr_data[0][timestep])):
                terms = []
                constant: float = 0
                for operand_nb in range(len(operands_linear_expr_data)):
                    constant += operands_linear_expr_data[operand_nb][timestep][
                        scenario
                    ].constant
                    terms.extend(
                        operands_linear_expr_data[operand_nb][timestep][scenario].terms
                    )
                linear_expr_data_given_time.append(
                    LinearExpressionData(terms=terms, constant=constant)
                )
            linear_expr_datas.append(linear_expr_data_given_time)
        return linear_expr_datas

    def multiplication(
        self, node: MultiplicationNode
    ) -> List[List[LinearExpressionData]]:
        lhs_all_time_scenario = visit(node.left, self)
        rhs_all_time_scenario = visit(node.right, self)
        linear_expr_datas = []
        for lhs_given_time, rhs_given_time in zip(
            lhs_all_time_scenario, rhs_all_time_scenario
        ):
            linear_expr_data_given_time = []
            for lhs, rhs in zip(lhs_given_time, rhs_given_time):
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
                linear_expr_data_given_time.append(actual_expr)
            linear_expr_datas.append(linear_expr_data_given_time)
        return linear_expr_datas

    def division(self, node: DivisionNode) -> List[List[LinearExpressionData]]:
        lhs_all_time_scenario = visit(node.left, self)
        rhs_all_time_scenario = visit(node.right, self)
        linear_expr_datas = []
        for lhs_given_time, rhs_given_time in zip(
            lhs_all_time_scenario, rhs_all_time_scenario
        ):
            linear_expr_data_given_time = []
            for lhs, rhs in zip(lhs_given_time, rhs_given_time):
                if rhs.terms:
                    raise ValueError(
                        "The second operand of a division must be a constant expression."
                    )
                divider = rhs.constant
                actual_expr = lhs
                actual_expr.constant /= divider
                for t in actual_expr.terms:
                    t.coefficient /= divider
                linear_expr_data_given_time.append(actual_expr)
            linear_expr_datas.append(linear_expr_data_given_time)
        return linear_expr_datas

    @staticmethod
    def _get_timestep(
        time_index: TimeIndex, current_timestep: Optional[int]
    ) -> Optional[int]:
        if isinstance(time_index, TimeShift):
            if current_timestep is None:
                raise ValueError("Cannot shift a time-independent expression.")
            return current_timestep + time_index.timeshift
        if isinstance(time_index, TimeStep):
            return time_index.timestep
        if isinstance(time_index, NoTimeIndex):
            return None
        else:
            raise TypeError(f"Type {type(time_index)} is not a valid TimeIndex type.")

    @staticmethod
    def _get_scenario(
        scenario_index: ScenarioIndex, current_scenario: int
    ) -> Optional[int]:
        if isinstance(scenario_index, OneScenarioIndex):
            return scenario_index.scenario
        elif isinstance(scenario_index, CurrentScenarioIndex):
            return current_scenario
        elif isinstance(scenario_index, NoScenarioIndex):
            return None
        else:
            raise TypeError(
                f"Type {type(scenario_index)} is not a valid ScenarioIndex type."
            )

    def literal(self, node: LiteralNode) -> List[List[LinearExpressionData]]:
        return [
            [
                LinearExpressionData([], node.value)
                for _ in self.linear_expr_scenario_indexing()
            ]
            for _ in self.linear_expr_time_indexing()
        ]

    def linear_expr_scenario_indexing(self) -> Iterable[int]:
        return self.scenarios if self.scenarios else [0]

    def linear_expr_time_indexing(self) -> Iterable[int]:
        return self.timesteps if self.timesteps else [0]

    def comparison(self, node: ComparisonNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Linear expression cannot contain a comparison operator.")

    def variable(self, node: VariableNode) -> List[List[LinearExpressionData]]:
        raise ValueError(
            "Variables need to be associated with their component ID before linearization."
        )

    def parameter(self, node: ParameterNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Parameters must be evaluated before linearization.")

    def comp_variable(
        self, node: ComponentVariableNode
    ) -> List[List[LinearExpressionData]]:
        raise ValueError(
            "Variables need to be associated with their timestep/scenario before linearization."
        )

    def pb_variable(
        self, node: ProblemVariableNode
    ) -> List[List[LinearExpressionData]]:
        return [
            [
                LinearExpressionData(
                    [
                        MutableTerm(
                            1,
                            node.component_id,
                            node.name,
                            time_index=self._get_timestep(
                                node.time_index, current_timestep
                            ),
                            scenario_index=self._get_scenario(
                                node.scenario_index, current_scenario
                            ),
                        )
                    ],
                    0,
                )
                for current_scenario in self.linear_expr_scenario_indexing()
            ]
            for current_timestep in self.linear_expr_time_indexing()
        ]

    def comp_parameter(
        self, node: ComponentParameterNode
    ) -> List[List[LinearExpressionData]]:
        raise ValueError(
            "Parameters need to be associated with their timestep/scenario before linearization."
        )

    def pb_parameter(
        self, node: ProblemParameterNode
    ) -> List[List[LinearExpressionData]]:
        # TODO SL: not the best place to do this.
        # in the future, we should evaluate coefficients of variables as time vectors once for all timesteps
        # TODO: Update the value_provider to be able to pass a list time/scenario indices
        linear_expr_datas = []
        for current_timestep in self.linear_expr_time_indexing():
            linear_expr_given_time = []
            time_index = self._get_timestep(node.time_index, current_timestep)
            for current_scenario in self.linear_expr_scenario_indexing():
                scenario_index = self._get_scenario(
                    node.scenario_index, current_scenario
                )
                linear_expr_given_time.append(
                    LinearExpressionData(
                        [],
                        self._value_provider().get_parameter_value(
                            node.component_id, node.name, time_index, scenario_index
                        ),
                    )
                )
            linear_expr_datas.append((linear_expr_given_time))
        return linear_expr_datas

    def time_eval(self, node: TimeEvalNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Time operators need to be expanded before linearization.")

    def time_shift(self, node: TimeShiftNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Time operators need to be expanded before linearization.")

    def time_sum(self, node: TimeSumNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Time operators need to be expanded before linearization.")

    def all_time_sum(self, node: AllTimeSumNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Time operators need to be expanded before linearization.")

    def _value_provider(self) -> ParameterGetter:
        if self.value_provider is None:
            raise ValueError(
                "A value provider must be specified to linearize a time operator node."
                " This is required in order to evaluate the value of potential parameters"
                " used to specified the time ids on which the time operator applies."
            )
        return self.value_provider

    def scenario_operator(
        self, node: ScenarioOperatorNode
    ) -> List[List[LinearExpressionData]]:
        raise ValueError("Scenario operators need to be expanded before linearization.")

    def port_field(self, node: PortFieldNode) -> List[List[LinearExpressionData]]:
        raise ValueError("Port fields must be replaced before linearization.")

    def port_field_aggregator(
        self, node: PortFieldAggregatorNode
    ) -> List[List[LinearExpressionData]]:
        raise ValueError(
            "Port fields aggregators must be replaced before linearization."
        )


def linearize_expression(
    expression: ExpressionNode,
    timesteps: Optional[Iterable[int]] = None,
    scenarios: Optional[Iterable[int]] = None,
    value_provider: Optional[ParameterGetter] = None,
) -> List[List[LinearExpression]]:
    linear_expr_datas = visit(
        expression,
        LinearExpressionBuilder(
            value_provider=value_provider, timesteps=timesteps, scenarios=scenarios
        ),
    )
    linear_expr = []
    for timestep in range(
        len(linear_expr_datas)
    ):  # Suppose timesteps are from 0 to len(linear_expr_datas), i.e. no expr on custom range, need Dict for that
        linear_expr_given_time = []
        for scenario in range(len(linear_expr_datas[timestep])):
            linear_expr_given_time.append(linear_expr_datas[timestep][scenario].build())
        linear_expr.append(linear_expr_given_time)

    return linear_expr
