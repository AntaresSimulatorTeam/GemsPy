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

import warnings

import pytest

from gems.expression.expression import (
    ExpressionNode,
    ScenarioOperatorNode,
    literal,
    param,
    port_field,
    var,
)
from gems.expression.indexing_structure import IndexingStructure
from gems.model import Constraint, float_variable, model
from gems.model.common import ValueType
from gems.model.port import port_field_def
from gems.model.variable import bool_var, int_variable


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
    For all t, p_min <= p[t] <= p_max * alpha[t] where p_min, p_max are literal parameters and alpha is an input timeseries
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
            objective_contributions={"operational": var("generation").variance()},
        )
    assert str(exc.value) == "Objective contribution must be a linear expression."


@pytest.mark.parametrize(
    "expression",
    [
        var("x") <= 0,
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


# --- Issue #76: tolerate absence of expec() in objective contributions ---


def test_objective_without_expec_on_scenario_var_emits_warning_and_wraps() -> None:
    """
    When a scenario-dependent variable is used in an objective contribution
    without expec(), the model() factory should auto-wrap with expec() and
    emit a UserWarning.
    """
    scenario_var = float_variable("generation", structure=IndexingStructure(True, True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = model(
            id="auto_wrap_model",
            variables=[scenario_var],
            objective_contributions={"operational": var("generation").time_sum()},
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "expec()" in str(user_warnings[0].message)
    assert "operational" in str(user_warnings[0].message)
    # The stored expression must now be wrapped in expec()
    stored = m.objective_contributions["operational"]
    assert isinstance(stored, ScenarioOperatorNode)
    assert stored.name == "Expectation"


def test_objective_with_explicit_expec_emits_no_warning() -> None:
    """
    When expec() is already present in the objective contribution,
    no warning should be emitted and the expression is not double-wrapped.
    """
    scenario_var = float_variable("generation", structure=IndexingStructure(True, True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = model(
            id="explicit_expec_model",
            variables=[scenario_var],
            objective_contributions={
                "operational": var("generation").time_sum().expec()
            },
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
    stored = m.objective_contributions["operational"]
    assert isinstance(stored, ScenarioOperatorNode)
    assert stored.name == "Expectation"
    # Must not be double-wrapped
    assert not isinstance(stored.operand, ScenarioOperatorNode)


def test_objective_with_non_scenario_var_emits_no_warning() -> None:
    """
    When the objective contribution is already a scalar (no scenario dimension),
    no auto-wrapping or warning should occur.
    """
    non_scenario_var = float_variable(
        "generation", structure=IndexingStructure(True, False)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(
            id="non_scenario_model",
            variables=[non_scenario_var],
            objective_contributions={"operational": var("generation").time_sum()},
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0


def test_objective_with_time_dimension_remaining_is_still_rejected() -> None:
    """
    Auto-wrapping only applies when scenario=True and time=False.
    If time dimension is still present (missing time_sum()), validation must fail.
    """
    scenario_var = float_variable("generation", structure=IndexingStructure(True, True))
    with pytest.raises(ValueError, match="real-valued expression"):
        model(
            id="bad_time_model",
            variables=[scenario_var],
            # No time_sum() — still has time dimension → rejected regardless
            objective_contributions={"operational": var("generation")},
        )


# --- Variable factory functions ---


def test_bool_var_has_binary_type_and_unit_bounds() -> None:
    """bool_var() creates a Variable with BINARY type and bounds [0, 1]."""
    v = bool_var("on_off")
    assert v.name == "on_off"
    assert v.data_type == ValueType.BINARY
    assert v.lower_bound == literal(0)
    assert v.upper_bound == literal(1)


def test_bool_var_default_structure_is_time_and_scenario() -> None:
    v = bool_var("flag")
    assert v.structure == IndexingStructure(True, True)


def test_int_variable_has_integer_type() -> None:
    """int_variable() creates a Variable with INTEGER type."""
    v = int_variable("count", lower_bound=literal(0), upper_bound=literal(10))
    assert v.data_type == ValueType.INTEGER
    assert v.lower_bound == literal(0)
    assert v.upper_bound == literal(10)


def test_variable_eq_with_non_variable_returns_false() -> None:
    """Variable.__eq__ returns False when compared with a non-Variable object."""
    v = float_variable("x")
    assert v != "x"
    assert v != 42
    assert v != None  # noqa: E711
