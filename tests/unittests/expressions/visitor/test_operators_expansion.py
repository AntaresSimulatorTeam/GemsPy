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

from dataclasses import dataclass
from typing import Dict

import pytest

from gems.expression import ExpressionNode, LiteralNode
from gems.expression.equality import expressions_equal
from gems.expression.expression import (
    CurrentScenarioIndex,
    NoScenarioIndex,
    NoTimeIndex,
    ProblemParameterNode,
    ProblemVariableNode,
    TimeShift,
    TimeStep,
    comp_param,
    comp_var,
    problem_param,
    problem_var,
)
from gems.expression.indexing import IndexingStructureProvider
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.operators_expansion import ProblemDimensions, expand_operators

P = comp_param("c", "p")
X = comp_var("c", "x")
CONST = comp_var("c", "const")


def shifted_P(t: int = 0) -> ProblemParameterNode:
    return problem_param("c", "p", TimeShift(t), CurrentScenarioIndex())


def P_at(t: int = 0) -> ProblemParameterNode:
    return problem_param("c", "p", TimeStep(t), CurrentScenarioIndex())


def X_at(t: int = 0) -> ProblemVariableNode:
    return problem_var("c", "x", TimeStep(t), CurrentScenarioIndex())


def shifted_X(t: int = 0) -> ProblemVariableNode:
    return problem_var("c", "x", TimeShift(t), CurrentScenarioIndex())


def const() -> ProblemVariableNode:
    return problem_var("c", "x", NoTimeIndex(), NoScenarioIndex())


def evaluate_literal(node: ExpressionNode) -> int:
    if isinstance(node, LiteralNode):
        return int(node.value)
    raise NotImplementedError("Can only evaluate literal nodes.")


@dataclass(frozen=True)
class AllTimeScenarioDependent(IndexingStructureProvider):
    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)

    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return IndexingStructure(True, True)


@dataclass(frozen=True)
class StructureProviderDict(IndexingStructureProvider):
    """
    Defines indexing structure through dictionaries. Default is time-scenario dependent.
    """

    variables: Dict[str, IndexingStructure]
    parameters: Dict[str, IndexingStructure]

    def get_parameter_structure(self, name: str) -> IndexingStructure:
        return self.parameters.get(name, IndexingStructure(True, True))

    def get_variable_structure(self, name: str) -> IndexingStructure:
        return self.variables.get(name, IndexingStructure(True, True))

    def get_component_variable_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return self.variables.get(name, IndexingStructure(True, True))

    def get_component_parameter_structure(
        self, component_id: str, name: str
    ) -> IndexingStructure:
        return self.parameters.get(name, IndexingStructure(True, True))


@pytest.mark.parametrize(
    "expr,expected",
    [
        (X.time_sum(), X_at(0) + X_at(1)),
        (X.shift(-1), shifted_X(-1)),
        (X.time_sum(-2, 0), shifted_X(-2) + (shifted_X(-1) + shifted_X(0))),
        ((P * X).shift(-1), shifted_P(-1) * shifted_X(-1)),
        (X.shift(-1).shift(+1), shifted_X(0)),
        (
            P * (P * X).time_sum(0, 1),
            shifted_P(0) * (shifted_P(0) * shifted_X(0) + shifted_P(1) * shifted_X(1)),
        ),
        (X.eval(2).time_sum(), X_at(2) + X_at(2)),
    ],
)
def test_operators_expansion(expr: ExpressionNode, expected: ExpressionNode) -> None:
    expanded = expand_operators(
        expr, ProblemDimensions(2, 1), evaluate_literal, AllTimeScenarioDependent()
    )
    assert expressions_equal(expanded, expected)


def test_time_scenario_independent_var_has_no_time_or_scenario_index():
    structure_provider = StructureProviderDict(
        parameters={}, variables={"const": IndexingStructure(False, False)}
    )
    expr = (X + CONST).time_sum()
    expanded = expand_operators(
        expr, ProblemDimensions(2, 1), evaluate_literal, structure_provider
    )
    assert expanded == X_at(0) + const() + X_at(1) + const()
