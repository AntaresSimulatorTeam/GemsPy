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


import time

from gems.expression import (
    AdditionNode,
    DivisionNode,
    LiteralNode,
    ParameterNode,
    VariableNode,
)
from gems.expression.copy import copy_expression
from gems.expression.equality import expressions_equal
from gems.expression.expression import (
    AllTimeSumNode,
    ComponentParameterNode,
    MultiplicationNode,
    TimeEvalNode,
    TimeShiftNode,
)


def test_copy_large_addition_is_linear() -> None:
    """
    Copying an AdditionNode with T operands must be O(T), not O(T²).

    Before the fix, CopyVisitor inherited the default addition() from
    ExpressionVisitorOperations which accumulated results with `res = res + o`.
    Each call to __add__ flattens the AdditionNode by copying the accumulated
    operand list, giving 1+2+...+(T-1) = O(T²) list copies in total.

    The fix overrides addition() in CopyVisitor to build AdditionNode directly
    from a list comprehension, reducing the cost to O(T).

    We verify linearity by checking that the time ratio between T=10_000 and
    T=1_000 stays below 20 (linear ≈ 10, quadratic ≈ 100).
    """
    small_n = 1_000
    large_n = 10_000

    small_node = AdditionNode([VariableNode(f"x{i}") for i in range(small_n)])
    large_node = AdditionNode([VariableNode(f"x{i}") for i in range(large_n)])

    t0 = time.perf_counter()
    copy_expression(small_node)
    small_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    copy_expression(large_node)
    large_time = time.perf_counter() - t0

    ratio = large_time / small_time
    assert ratio < 20, (
        f"copy_expression scaling looks super-linear: "
        f"T={small_n} took {small_time:.4f}s, "
        f"T={large_n} took {large_time:.4f}s, "
        f"ratio={ratio:.1f} (expected <20 for O(T))"
    )


def test_copy_ast() -> None:
    ast = AllTimeSumNode(
        DivisionNode(
            TimeEvalNode(
                AdditionNode([LiteralNode(1), VariableNode("x")]), ParameterNode("p")
            ),
            TimeShiftNode(
                MultiplicationNode(LiteralNode(1), VariableNode("x")),
                ComponentParameterNode("comp1", "p"),
            ),
        ),
    )
    copy = copy_expression(ast)
    assert expressions_equal(ast, copy)
