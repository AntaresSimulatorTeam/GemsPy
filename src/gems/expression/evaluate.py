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
from typing import Dict, Iterable

import numpy as np
import pandas as pd

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
from gems.expression.utils import ProblemDimensions

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
    def get_variable_value(self, name: str) -> pd.DataFrame: ...

    @abstractmethod
    def get_parameter_value(self, name: str) -> pd.DataFrame: ...

    @abstractmethod
    def get_component_variable_value(
        self, component_id: str, name: str
    ) -> pd.DataFrame: ...

    @abstractmethod
    def get_component_parameter_value(
        self, component_id: str, name: str
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class EvaluationContext(ValueProvider):
    """
    Simple value provider relying on dictionaries.
    Does not support component variables/parameters.
    """

    variables: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)

    def get_variable_value(self, name: str) -> pd.DataFrame:
        return self.variables[name]

    def get_parameter_value(self, name: str) -> pd.DataFrame:
        return self.parameters[name]

    def get_component_variable_value(
        self, component_id: str, name: str
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def get_component_parameter_value(
        self, component_id: str, name: str
    ) -> pd.DataFrame:
        raise NotImplementedError()


@dataclass(frozen=True)
class EvaluationVisitor(ExpressionVisitorOperations[pd.DataFrame]):
    """
    Evaluates the expression with respect to the provided context
    (variables and parameters values).
    """

    context: ValueProvider
    timesteps_count: int
    scenarios_count: int

    def literal(self, node: LiteralNode) -> pd.DataFrame:
        return pd.DataFrame(
            np.full((self.timesteps_count * self.scenarios_count, 1), node.value),
            index=pd.MultiIndex.from_product(
                [[0], [0], range(self.timesteps_count), range(self.scenarios_count)],
                names=["timeshift", "scenarioshift", "timestep", "scenario"],
            ),
            columns=["value"],
        )

    def comparison(self, node: ComparisonNode) -> pd.DataFrame:
        raise ValueError("Cannot evaluate comparison operator.")

    def variable(self, node: VariableNode) -> pd.DataFrame:
        return self.context.get_variable_value(node.name)

    def parameter(self, node: ParameterNode) -> pd.DataFrame:
        return self.context.get_parameter_value(node.name)

    def comp_parameter(self, node: ComponentParameterNode) -> pd.DataFrame:
        return self.context.get_component_parameter_value(node.component_id, node.name)

    def comp_variable(self, node: ComponentVariableNode) -> pd.DataFrame:
        return self.context.get_component_variable_value(node.component_id, node.name)

    def pb_parameter(self, node: ProblemParameterNode) -> pd.DataFrame:
        raise ValueError("Should not reach here.")

    def pb_variable(self, node: ProblemVariableNode) -> pd.DataFrame:
        raise ValueError("Should not reach here.")

    def time_shift(self, node: TimeShiftNode) -> pd.DataFrame:
        raise NotImplementedError()

    def time_eval(self, node: TimeEvalNode) -> pd.DataFrame:
        raise NotImplementedError()

    def time_sum(self, node: TimeSumNode) -> pd.DataFrame:
        raise NotImplementedError()

    def all_time_sum(self, node: AllTimeSumNode) -> pd.DataFrame:
        raise NotImplementedError()

    def scenario_operator(self, node: ScenarioOperatorNode) -> pd.DataFrame:
        raise NotImplementedError()

    def port_field(self, node: PortFieldNode) -> pd.DataFrame:
        raise NotImplementedError()

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> pd.DataFrame:
        raise NotImplementedError()


def evaluate(
    expression: ExpressionNode,
    value_provider: ValueProvider,
    dimensions: ProblemDimensions,
) -> pd.DataFrame:
    return visit(
        expression,
        EvaluationVisitor(
            value_provider, dimensions.timesteps_count, dimensions.scenarios_count
        ),
    )
