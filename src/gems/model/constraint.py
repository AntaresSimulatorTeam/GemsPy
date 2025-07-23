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

from dataclasses import dataclass, field, replace
from typing import Any

from gems.expression.degree import is_constant
from gems.expression.equality import expressions_equal, expressions_equal_if_present
from gems.expression.expression import (
    Comparator,
    ComparisonNode,
    ExpressionNode,
    is_non_negative,
    is_unbounded,
    literal,
)
from gems.expression.print import print_expr
from gems.model.common import ProblemContext


@dataclass
class Constraint:
    """
    A constraint linking variables and parameters of a model together.

    No variable is expected on the right hand side of the constraint.
    """

    name: str
    expression: ExpressionNode
    lower_bound: ExpressionNode = field(default=literal(-float("inf")))
    upper_bound: ExpressionNode = field(default=literal(float("inf")))
    context: ProblemContext = field(default=ProblemContext.OPERATIONAL)

    def __post_init__(
        self,
    ) -> None:
        if isinstance(self.expression, ComparisonNode):
            if not is_unbounded(self.lower_bound) or not is_unbounded(self.upper_bound):
                raise ValueError(
                    "Both comparison between two expressions and a bound are specfied, set either only a comparison between expressions or a single linear expression with bounds."
                )

            if self.expression.comparator == Comparator.LESS_THAN:
                # lhs - rhs <= 0
                self.upper_bound = literal(0)
                self.lower_bound = literal(-float("inf"))
            elif self.expression.comparator == Comparator.GREATER_THAN:
                # lhs - rhs >= 0
                self.lower_bound = literal(0)
                self.upper_bound = literal(float("inf"))
            else:  # lhs - rhs == 0
                self.lower_bound = literal(0)
                self.upper_bound = literal(0)

            self.expression = self.expression.left - self.expression.right

        else:
            for bound in [self.lower_bound, self.upper_bound]:
                if not is_constant(bound):
                    raise ValueError(
                        f"The bounds of a constraint should not contain variables, {print_expr(bound)} was given."
                    )

            if is_unbounded(self.lower_bound) and is_non_negative(self.lower_bound):
                raise ValueError("Lower bound should not be +Inf")

            if is_unbounded(self.upper_bound) and not is_non_negative(self.upper_bound):
                raise ValueError("Upper bound should not be -Inf")

    def replicate(self, /, **changes: Any) -> "Constraint":
        return replace(self, **changes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Constraint):
            return False
        return (
            self.name == other.name
            and expressions_equal(self.expression, other.expression)
            and expressions_equal_if_present(self.lower_bound, other.lower_bound)
            and expressions_equal_if_present(self.upper_bound, other.upper_bound)
        )
