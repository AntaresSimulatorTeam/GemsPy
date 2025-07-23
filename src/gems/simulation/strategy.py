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
from typing import Generator, Optional

from gems.expression import ExpressionNode, literal
from gems.model import Constraint, Model, ProblemContext, Variable


class ModelSelectionStrategy(ABC):
    """
    Abstract class to specify the strategy of the created problem.
    Its derived classes select variables and constraints for the optimization problems:
        - InvestmentProblemStrategy: Keep investment and coupling variables and constraints only for a BendersDecomposed master
        - OperationalProblemStrategy: Keep operational and coupling variables and constraints only for a BendersDecomposed sub-problems
        - MergedProblemStrategy: Keep all variables and constraints
    """

    def get_variables(self, model: Model) -> Generator[Variable, None, None]:
        for variable in model.variables.values():
            if self._keep_from_context(variable.context):
                yield variable

    def get_constraints(self, model: Model) -> Generator[Constraint, None, None]:
        for constraint in model.get_all_constraints():
            if self._keep_from_context(constraint.context):
                yield constraint

    @abstractmethod
    def _keep_from_context(self, context: ProblemContext) -> bool:
        ...

    @abstractmethod
    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        ...


class MergedProblemStrategy(ModelSelectionStrategy):
    def _keep_from_context(self, context: ProblemContext) -> bool:
        return True

    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        yield model.objective_operational_contribution
        yield model.objective_investment_contribution


class InvestmentProblemStrategy(ModelSelectionStrategy):
    def _keep_from_context(self, context: ProblemContext) -> bool:
        return (
            context == ProblemContext.INVESTMENT or context == ProblemContext.COUPLING
        )

    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        yield model.objective_investment_contribution


class OperationalProblemStrategy(ModelSelectionStrategy):
    def _keep_from_context(self, context: ProblemContext) -> bool:
        return (
            context == ProblemContext.OPERATIONAL or context == ProblemContext.COUPLING
        )

    def get_objectives(
        self, model: Model
    ) -> Generator[Optional[ExpressionNode], None, None]:
        yield model.objective_operational_contribution


class RiskManagementStrategy(ABC):
    """
    Abstract functor class for risk management
    Its derived classes will implement risk measures:
        - UniformRisk   : The default case. All expressions have the same weight
        - ExpectedValue : Computes the product prob * expression
    TODO For now, it will only take into account the Expected Value
    TODO In the future could have other risk measures?
    """

    def __call__(self, expr: ExpressionNode) -> ExpressionNode:
        return self._modify_expression(expr)

    @abstractmethod
    def _modify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        ...


class UniformRisk(RiskManagementStrategy):
    def _modify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        return expr


class ExpectedValue(RiskManagementStrategy):
    def __init__(self, prob: float) -> None:
        self._prob = prob

    def _modify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        return literal(self._prob) * expr
