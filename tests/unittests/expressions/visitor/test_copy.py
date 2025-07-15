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
