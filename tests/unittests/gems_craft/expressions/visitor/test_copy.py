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

from gems_craft.expression import (
    AdditionNode,
    DivisionNode,
    LiteralNode,
    ParameterNode,
    VariableNode,
)
from gems_craft.expression.copy import copy_expression
from gems_craft.expression.equality import expressions_equal
from gems_craft.expression.expression import (
    AllTimeSumNode,
    MultiplicationNode,
    SetIndexNode,
    SumOverNode,
    TimeEvalNode,
    TimeShiftNode,
)


def test_copy_ast() -> None:
    ast = AllTimeSumNode(
        DivisionNode(
            TimeEvalNode(
                AdditionNode([LiteralNode(1), VariableNode("x")]), ParameterNode("p")
            ),
            TimeShiftNode(
                MultiplicationNode(LiteralNode(1), VariableNode("x")),
                ParameterNode("p"),
            ),
        ),
    )
    copy = copy_expression(ast)
    assert expressions_equal(ast, copy)


def test_copy_set_index_sum_over() -> None:
    bare = SetIndexNode(VariableNode("x"), "fuel")
    assert expressions_equal(bare, copy_expression(bare))

    positioned = SetIndexNode(VariableNode("x"), "fuel", position=LiteralNode(2))
    assert expressions_equal(positioned, copy_expression(positioned))

    shifted = SetIndexNode(VariableNode("x"), "fuel", relative_shift=LiteralNode(1))
    assert expressions_equal(shifted, copy_expression(shifted))

    aggregated = SumOverNode(VariableNode("x"), "fuel")
    assert expressions_equal(aggregated, copy_expression(aggregated))
