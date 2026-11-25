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

"""A time shift, time index or time sum bound must be time-independent and
contain no variable, port field or time operator, and a time shift, time index
or time sum with bounds must apply to an expression that depends on time,
checked when the model is created (here through the Python API)."""

import re
from typing import Any, Dict, Optional

import pytest

from gems_craft.expression.expression import (
    ExpressionNode,
    param,
    port_field,
    relative_time,
    var,
)
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model import Constraint, Model, float_parameter, float_variable, model

TIME_DEPENDENT = IndexingStructure(True, False)
CONSTANT = IndexingStructure(False, False)
SCENARIO_DEPENDENT = IndexingStructure(False, True)

_DOES_NOT_DEPEND_ON_TIME = "reads an expression that does not depend on time."


def _model(
    constraint: Optional[ExpressionNode] = None,
    extra_outputs: Optional[Dict[str, ExpressionNode]] = None,
    objective: Optional[ExpressionNode] = None,
    upper_bound: Optional[ExpressionNode] = None,
) -> Model:
    return model(
        id="m",
        parameters=[
            float_parameter("k", TIME_DEPENDENT),
            float_parameter("j", TIME_DEPENDENT),
            float_parameter("c", CONSTANT),
            float_parameter("s", SCENARIO_DEPENDENT),
        ],
        variables=[
            float_variable("x"),
            float_variable("y", upper_bound=upper_bound),
            float_variable("z", structure=CONSTANT),
        ],
        constraints=(
            [Constraint("c0", constraint <= 10)] if constraint is not None else None
        ),
        extra_outputs=extra_outputs,
        objective_contributions=({"o": objective} if objective is not None else None),
    )


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(var("x").time_sum_between(0, param("c")), id="constant"),
        pytest.param(var("x").time_sum_between(0, param("s")), id="scenario-dependent"),
        pytest.param(var("x").shift(param("s")), id="shift-scenario-dependent"),
        pytest.param(var("x").eval(param("c")), id="index-constant"),
        # sum(z) accepts an expression that does not depend on time.
        pytest.param(var("z").time_sum(), id="sum-of-constant"),
        pytest.param((var("x") + var("z")).shift(-1), id="shift-of-time-dependent"),
    ],
)
def test_valid_time_operators_are_accepted(expr: ExpressionNode) -> None:
    _model(constraint=expr)


@pytest.mark.parametrize(
    "expr, error",
    [
        pytest.param(
            var("x").time_sum_between(relative_time(-param("k")), relative_time(0)),
            "the time sum bound '(t + -(k))' depends on time",
            id="relative-offset",
        ),
        pytest.param(
            var("x").time_sum_between(0, param("k") + param("c") + param("j")),
            "the time sum bound '(k + c + j)' depends on time, but must be "
            "time-independent. Declare 'j', 'k' with `time-dependent: false`.",
            id="names-the-time-dependent-parameters",
        ),
        pytest.param(
            var("x").time_sum_between(0, param("c").eval(0)),
            "the time sum bound '(c.eval(0.0))' cannot contain a time operator.",
            id="time-operator",
        ),
        pytest.param(
            var("x").shift(-param("k")),
            "the time shift '-(k)' depends on time, but must be time-independent. "
            "Declare 'k' with `time-dependent: false`.",
            id="shift",
        ),
        pytest.param(
            var("x").eval(param("k")),
            "the time index 'k' depends on time, but must be time-independent. "
            "Declare 'k' with `time-dependent: false`.",
            id="index",
        ),
        pytest.param(
            var("x").time_sum_between(0, var("y")),
            "the time sum bound 'y' cannot contain a variable.",
            id="variable",
        ),
        pytest.param(
            var("x").time_sum_between(0, port_field("p", "f")),
            "the time sum bound 'p.f' cannot contain a port field.",
            id="port-field",
        ),
        # A time shift or time index of an expression that does not depend on time
        pytest.param(
            var("z").shift(0),
            f"the time shift of 'z' {_DOES_NOT_DEPEND_ON_TIME}",
            id="z[t]",
        ),
        pytest.param(
            var("z").shift(-1),
            f"the time shift of 'z' {_DOES_NOT_DEPEND_ON_TIME}",
            id="z[t-1]",
        ),
        pytest.param(
            var("z").eval(0),
            f"the time index of 'z' {_DOES_NOT_DEPEND_ON_TIME}",
            id="z[0]",
        ),
        pytest.param(
            param("c").shift(-1),
            f"the time shift of 'c' {_DOES_NOT_DEPEND_ON_TIME}",
            id="c[t-1]",
        ),
        pytest.param(
            var("z").time_sum(-1, 0),
            f"the time sum of 'z' {_DOES_NOT_DEPEND_ON_TIME}",
            id="sum(t-1 .. t, z)",
        ),
    ],
)
def test_invalid_time_operators_are_rejected_in_constraints(
    expr: ExpressionNode, error: str
) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"Model 'm', constraint 'c0': {error}")
    ):
        _model(constraint=expr)


@pytest.mark.parametrize(
    "location, kwargs",
    [
        pytest.param(
            "extra output 'e'",
            {"extra_outputs": {"e": var("z").shift(-1)}},
            id="extra-output",
        ),
        pytest.param(
            "objective contribution 'o'",
            {"objective": var("z").shift(-1).expec()},
            id="objective",
        ),
        pytest.param(
            "variable 'y'",
            {"upper_bound": param("k").shift(param("j"))},
            id="variable-bound",
        ),
    ],
)
def test_invalid_time_operators_are_rejected_everywhere(
    location: str, kwargs: Dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=re.escape(f"Model 'm', {location}: ")):
        _model(**kwargs)
