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

from andromede.expression import literal, param, var
from andromede.model import model
from andromede.model.constraint import Constraint
from andromede.thermal_heuristic.model import ModelEditor


def test_variable_in_expression() -> None:
    model_editer = ModelEditor(model("model"))

    expression_1 = var("x").sum() + var("y")
    expression_2 = (
        var("z").shift(-1) + var("z") + literal(0) * var("y") + var("x").expec()
    )

    assert model_editer.variable_in_expression(expression_1, ["x"]) is True
    assert model_editer.variable_in_expression(expression_1, ["y"]) is True
    assert model_editer.variable_in_expression(expression_1, ["z"]) is False
    assert model_editer.variable_in_expression(expression_2, ["x"]) is True
    assert model_editer.variable_in_expression(expression_2, ["y"]) is True
    assert model_editer.variable_in_expression(expression_2, ["z"]) is True


def test_variable_in_constraint() -> None:
    model_editer = ModelEditor(model("model"))

    expression_1 = var("x").sum() + var("y")
    expression_2 = (
        var("z").shift(-1) + var("z") + literal(0) * var("y") + var("x").expec()
    )
    expression_3 = var("y") - var("y") + param("a")
    constraint = Constraint("cst", expression_1 <= expression_2 <= expression_3)

    assert model_editer.variable_in_constraint(constraint, ["x"]) is True
    assert model_editer.variable_in_constraint(constraint, ["y"]) is True
    assert model_editer.variable_in_constraint(constraint, ["z"]) is True
    assert model_editer.variable_in_constraint(constraint, ["xy"]) is False
    assert model_editer.variable_in_constraint(constraint, ["a"]) is True
