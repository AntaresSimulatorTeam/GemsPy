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
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        timesteps: Optional[Iterable[Tuple[int, int]]],  # current_timestep, timeshift
        scenarios: Optional[
            Iterable[Tuple[int, int]]
        ],  # current_scenario, scenarioshift
    ) -> pd.DataFrame:
        pass


@dataclass
class MutableTerm:
    coefficient: Union[pd.DataFrame, float]
    component_id: str
    variable_name: str
    # time_index: Optional[int]
    # scenario_index: Optional[int]

    def to_key(self) -> TermKey:
        return TermKey(
            self.component_id,
            self.variable_name,
            # self.time_index,
            # self.scenario_index,
        )

    def to_term(self) -> Term:
        return Term(
            self.coefficient,
            self.component_id,
            self.variable_name,
            # self.time_index,
            # self.scenario_index,
        )


@dataclass
class LinearExpressionData:
    terms: List[MutableTerm]
    constant: Union[pd.DataFrame, float]

    def build(self) -> LinearExpression:
        res_terms: Dict[TermKey, Any] = {}
        for t in self.terms:
            k = t.to_key()
            if k in res_terms:
                current_t = res_terms[k]
                if isinstance(current_t.coefficient, (int, float)) or isinstance(
                    t.coefficient, (int, float)
                ):
                    current_t.coefficient += t.coefficient
                else:
                    current_t.coefficient = current_t.coefficient.add(
                        t.coefficient, fill_value=0
                    )
            else:
                res_terms[k] = t
        for k, v in res_terms.items():
            res_terms[k] = v.to_term()
        return LinearExpression(res_terms, self.constant)


@dataclass(frozen=True)
class TimeScenarioShift:
    """
    A class to represent a time and scenario shift in dataframes of coefficient and constant linear expressions.

    A data value at line (t,w) for the x coefficient in column TimeScenarioShift(t', w') means in contraint number (t,w), there is the term value * x[t+t', w+w']
    """

    time_shift: int
    scenario_shift: int

    def __str__(self) -> str:
        return f"TimeScenarioShift(time_shift={self.time_shift}, scenario_shift={self.scenario_shift})"


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

    timesteps: Optional[Iterable[int]] = None
    scenarios: Optional[Iterable[int]] = None
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
        constant = 0
        for o in operands:
            if isinstance(o.constant, (int, float)) or isinstance(
                constant, (int, float)
            ):
                constant += o.constant
            else:
                # Creates a copy of the dataframe and reassign
                constant = constant.add(o.constant, fill_value=0)
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

        if isinstance(multiplier, (int, float)):
            actual_expr.constant *= multiplier
            for t in actual_expr.terms:
                t.coefficient *= multiplier
        else:
            # The shift dimensions are useless for constant as the value provider that has beforehand evaluated it has taken shifts into account to put the correct value for the expression indexed at (timestep, scenario).
            multiplier.set_index(
                multiplier.index.droplevel(["timeshift", "scenarioshift"]),
                inplace=True,
            )
            constant_df = multiplier.join(
                actual_expr.constant, how="inner", lsuffix="_multiplier"
            )
            constant_df["value"] = (
                constant_df["value"] * constant_df["value_multiplier"]
            )
            constant_df.drop("value_multiplier", axis=1, inplace=True)
            constant_df.index = constant_df.index.reorder_levels(
                ["timeshift", "scenarioshift", "timestep", "scenario"]
            )
            actual_expr.constant = constant_df
            for t in actual_expr.terms:
                coeff_df = multiplier.join(
                    t.coefficient, how="inner", lsuffix="_multiplier"
                )
                coeff_df["value"] = coeff_df["value"] * coeff_df["value_multiplier"]
                coeff_df.drop("value_multiplier", axis=1, inplace=True)
                coeff_df = coeff_df.reorder_levels(
                    ["timeshift", "scenarioshift", "timestep", "scenario"]
                )
                t.coefficient = coeff_df
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
        if isinstance(divider, (int, float)):
            actual_expr.constant /= divider
            for t in actual_expr.terms:
                t.coefficient /= divider
        else:
            divider.set_index(
                divider.index.droplevel(["timeshift", "scenarioshift"]),
                inplace=True,
            )
            constant_df = divider.join(
                actual_expr.constant, how="inner", lsuffix="_divider"
            )
            constant_df["value"] = constant_df["value"] / constant_df["value_divider"]
            constant_df.drop("value_divider", axis=1, inplace=True)
            constant_df.index = constant_df.index.reorder_levels(
                ["timeshift", "scenarioshift", "timestep", "scenario"]
            )
            actual_expr.constant = constant_df
            for t in actual_expr.terms:
                coeff_df = divider.join(t.coefficient, how="inner", lsuffix="_divider")
                coeff_df["value"] = coeff_df["value"] / coeff_df["value_divider"]
                coeff_df.drop("value_divider", axis=1, inplace=True)
                coeff_df = coeff_df.reorder_levels(
                    ["timeshift", "scenarioshift", "timestep", "scenario"]
                )
            t.coefficient = coeff_df
        return actual_expr

    @staticmethod
    def _get_timeshift(
        time_index: TimeIndex, current_timestep: Optional[int]
    ) -> Optional[int]:
        if isinstance(time_index, TimeShift):
            # if current_timestep is None:
            #     raise ValueError("Cannot shift a time-independent expression.")
            return time_index.timeshift
        if isinstance(time_index, TimeStep):
            return time_index.timestep - current_timestep
        if isinstance(time_index, NoTimeIndex):
            return None
        else:
            raise TypeError(f"Type {type(time_index)} is not a valid TimeIndex type.")

    @staticmethod
    def _get_scenarioshift(
        scenario_index: ScenarioIndex, current_scenario: int
    ) -> Optional[int]:
        if isinstance(scenario_index, OneScenarioIndex):
            return scenario_index.scenario - current_scenario
        elif isinstance(scenario_index, CurrentScenarioIndex):
            return 0
        elif isinstance(scenario_index, NoScenarioIndex):
            return None
        else:
            raise TypeError(
                f"Type {type(scenario_index)} is not a valid ScenarioIndex type."
            )

    @staticmethod
    def add_shift_indices(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df], keys=[(0, 0)], names=["timeshift", "scenarioshift"])

    def constant_value_data(
        self, value: float, indexing: Optional[pd.MultiIndex] = None
    ) -> float:
        if indexing is not None:
            return pd.DataFrame(
                np.full((len(indexing), 1), value),
                index=indexing,
                columns=["value"],
            )
        else:
            return value

    def literal(self, node: LiteralNode) -> LinearExpressionData:
        return LinearExpressionData(
            [],
            self.constant_value_data(node.value),
        )

    def timesteps_count(self) -> int:
        return len(self.timesteps) if self.timesteps else 1

    def scenarios_count(self) -> int:
        return len(self.scenarios) if self.scenarios else 1

    def linear_expr_time_indexing(self) -> Iterable[int]:

        return self.timesteps if self.timesteps else [0]

    def linear_expr_scenario_indexing(self) -> Iterable[int]:
        return self.scenarios if self.scenarios else [0]

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
        if isinstance(node.time_index, NoTimeIndex):
            time_indices = [
                (current_timestep, -current_timestep)
                for current_timestep in self.linear_expr_time_indexing()
            ]
        else:
            time_indices = [
                (
                    current_timestep,
                    self._get_timeshift(node.time_index, current_timestep),
                )
                for current_timestep in self.linear_expr_time_indexing()
            ]
        if isinstance(node.scenario_index, NoScenarioIndex):
            scenario_indices = [
                (current_scenario, -current_scenario)
                for current_scenario in self.linear_expr_scenario_indexing()
            ]
        else:
            scenario_indices = [
                (
                    current_scenario,
                    self._get_scenarioshift(node.scenario_index, current_scenario),
                )
                for current_scenario in self.linear_expr_scenario_indexing()
            ]
        index_map_values = [
            (
                timeshift if timeshift is not None else 0,
                scenarioshift if scenarioshift is not None else 0,
                timestep,
                scenario,
            )
            for (timestep, timeshift), (scenario, scenarioshift) in product(
                time_indices, scenario_indices
            )
        ]
        index = pd.MultiIndex.from_tuples(
            index_map_values,
            names=["timeshift", "scenarioshift", "timestep", "scenario"],
        )

        return LinearExpressionData(
            [
                MutableTerm(
                    self.constant_value_data(1, index),
                    node.component_id,
                    node.name,
                )
            ],
            self.constant_value_data(0, index),
        )

    def comp_parameter(self, node: ComponentParameterNode) -> LinearExpressionData:
        raise ValueError(
            "Parameters need to be associated with their timestep/scenario before linearization."
        )

    def pb_parameter(self, node: ProblemParameterNode) -> LinearExpressionData:
        time_indices = [
            (current_timestep, self._get_timeshift(node.time_index, current_timestep))
            for current_timestep in self.linear_expr_time_indexing()
        ]
        scenario_indices = [
            (
                current_scenario,
                self._get_scenarioshift(node.scenario_index, current_scenario),
            )
            for current_scenario in self.linear_expr_scenario_indexing()
        ]
        return LinearExpressionData(
            [],
            self._value_provider().get_parameter_value(
                node.component_id, node.name, time_indices, scenario_indices
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
    timesteps: Optional[Iterable[int]] = None,
    scenarios: Optional[Iterable[int]] = None,
    value_provider: Optional[ParameterGetter] = None,
) -> LinearExpression:
    return visit(
        expression,
        LinearExpressionBuilder(
            value_provider=value_provider, timesteps=timesteps, scenarios=scenarios
        ),
    ).build()
