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

import pytest

from gems.expression.expression import (
    ExpressionNode,
    comp_param,
    comp_var,
    literal,
    param,
    port_field,
    var,
)
from gems.model import Constraint, float_variable, model
from gems.model.port import port_field_def


@pytest.mark.parametrize(
    "name, expression, lb, ub, exp_name, exp_expr, exp_lb, exp_ub",
    [
        (
            "my_constraint",
            2 * var("my_var"),
            literal(5),
            literal(10),
            "my_constraint",
            2 * var("my_var"),
            literal(5),
            literal(10),
        ),
        (
            "my_constraint",
            2 * var("my_var"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var"),
            literal(-float("inf")),
            literal(float("inf")),
        ),
        (
            "my_constraint",
            2 * var("my_var") <= param("p"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var") - param("p"),
            literal(-float("inf")),
            literal(0),
        ),
        (
            "my_constraint",
            2 * var("my_var") >= param("p"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var") - param("p"),
            literal(0),
            literal(float("inf")),
        ),
        (
            "my_constraint",
            2 * var("my_var") == param("p"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var") - param("p"),
            literal(0),
            literal(0),
        ),
        (
            "my_constraint",
            2 * var("my_var").expec() == param("p"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var").expec() - param("p"),
            literal(0),
            literal(0),
        ),
        (
            "my_constraint",
            2 * var("my_var").shift(-1) == param("p"),
            literal(-float("inf")),
            literal(float("inf")),
            "my_constraint",
            2 * var("my_var").shift(-1) - param("p"),
            literal(0),
            literal(0),
        ),
    ],
)
def test_constraint_instantiation(
    name: str,
    expression: ExpressionNode,
    lb: ExpressionNode,
    ub: ExpressionNode,
    exp_name: str,
    exp_expr: ExpressionNode,
    exp_lb: ExpressionNode,
    exp_ub: ExpressionNode,
) -> None:
    constraint = Constraint(name, expression, lb, ub)
    assert constraint.name == exp_name
    assert constraint.expression == exp_expr
    assert constraint.lower_bound == exp_lb
    assert constraint.upper_bound == exp_ub


def test_if_both_comparison_expression_and_bound_given_for_constraint_init_then_it_should_raise_a_value_error() -> (
    None
):
    with pytest.raises(ValueError) as exc:
        Constraint("my_constraint", 2 * var("my_var") == param("my_param"), literal(2))
    assert (
        str(exc.value)
        == "Both comparison between two expressions and a bound are specfied, set either only a comparison between expressions or a single linear expression with bounds."
    )


def test_if_a_bound_is_not_constant_then_it_should_raise_a_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        Constraint("my_constraint", 2 * var("my_var"), var("x"))
    assert (
        str(exc.value)
        == "The bounds of a constraint should not contain variables, x was given."
    )


def test_writing_p_min_max_constraint_should_represent_all_expected_constraints() -> (
    None
):
    """
    Aim at representing the following mathematical constraints:
    For all t, p_min <= p[t] <= p_max * alpha[t] where p_min, p_max are literal paramters and alpha is an input timeseries
    """
    try:
        p_min = literal(5)
        p_max = literal(10)
        p = var("p")

        alpha = param("alpha")

        _ = Constraint("generation bounds", p, p_min, p_max * alpha)

    # Later on, the goal is to assert that when this constraint is sent to the solver, it correctly builds: for all t, p_min <= p[k,t,w] <= p_max * alpha[k,t,w]

    except Exception as exc:
        assert False, f"Writing p_min and p_max constraints raises an exception: {exc}"


def test_writing_min_up_constraint_should_represent_all_expected_constraints() -> None:
    """
    Aim at representing the following mathematical constraints:
    For all t, for all t' in [t+1, t+d_min_up], off_on[k,t,w] <= on[k,t',w]
    """
    try:
        d_min_up = literal(3)
        off_on = var("off_on")
        on = var("on")

        _ = Constraint(
            "min_up_time",
            off_on <= on.time_sum(literal(1), d_min_up),
        )

        # Later on, the goal is to assert that when this constraint is sent to the solver, it correctly builds: for all t, for all t' in [t+1, t+d_min_up], off_on[k,t,w] <= on[k,t',w]

    except Exception as exc:
        assert False, f"Writing min_up constraints raises an exception: {exc}"


def test_instantiating_a_model_with_non_linear_scenario_operator_in_the_objective_should_raise_type_error() -> (
    None
):
    with pytest.raises(ValueError) as exc:
        _ = model(
            id="model_with_non_linear_op",
            variables=[float_variable("generation")],
            objective_operational_contribution=var("generation").variance(),
        )
    assert str(exc.value) == "Objective contribution must be a linear expression."


@pytest.mark.parametrize(
    "expression",
    [
        var("x") <= 0,
        comp_var("c", "x"),
        comp_param("c", "x"),
        port_field("p", "f"),
        port_field("p", "f").sum_connections(),
    ],
)
def test_invalid_port_field_definition_should_raise(expression: ExpressionNode) -> None:
    with pytest.raises(ValueError) as exc:
        port_field_def(port_name="p", field_name="f", definition=expression)


def test_constraint_equals() -> None:
    # checks in particular that expressions are correctly compared
    assert Constraint(name="c", expression=var("x") <= param("p")) == Constraint(
        name="c", expression=var("x") <= param("p")
    )
    assert Constraint(name="c", expression=var("x") <= param("p")) != Constraint(
        name="c", expression=var("y") <= param("p")
    )
