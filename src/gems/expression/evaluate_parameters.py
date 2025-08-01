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

from gems.expression.evaluate import ValueProvider

from .copy import CopyVisitor
from .expression import (
    ComponentParameterNode,
    ExpressionNode,
    LiteralNode,
    ParameterNode,
)
from .visitor import visit


class ParameterValueProvider(ABC):
    @abstractmethod
    def get_parameter_value(self, name: str) -> float:
        ...

    @abstractmethod
    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        ...


@dataclass(frozen=True)
class ParameterResolver(CopyVisitor):
    """
    Duplicates the AST with replacement of parameter nodes by literal nodes.
    """

    context: ParameterValueProvider

    def parameter(self, node: ParameterNode) -> ExpressionNode:
        value: float = self.context.get_parameter_value(node.name)
        return LiteralNode(value)

    def comp_parameter(self, node: ComponentParameterNode) -> ExpressionNode:
        value: float = self.context.get_component_parameter_value(
            node.component_id, node.name
        )
        return LiteralNode(value)


def resolve_parameters(
    expression: ExpressionNode, parameter_provider: ParameterValueProvider
) -> ExpressionNode:
    return visit(expression, ParameterResolver(parameter_provider))
