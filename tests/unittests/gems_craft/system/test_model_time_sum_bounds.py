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

"""A time sum bound must be declared time-independent and contain no variable,
checked when the model is created (here through the Python API)."""

import re
from typing import Dict, Optional

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


def _model(
    constraint: Optional[ExpressionNode] = None,
    extra_outputs: Optional[Dict[str, ExpressionNode]] = None,
    objective: Optional[ExpressionNode] = None,
) -> Model:
    return model(
        id="m",
        parameters=[
            float_parameter("k", TIME_DEPENDENT),
            float_parameter("j", TIME_DEPENDENT),
            float_parameter("c", CONSTANT),
            float_parameter("s", SCENARIO_DEPENDENT),
        ],
        variables=[float_variable("x"), float_variable("y")],
        constraints=(
            [Constraint("c0", constraint <= 10)] if constraint is not None else None
        ),
        extra_outputs=extra_outputs,
        objective_contributions=({"o": objective} if objective is not None else None),
    )


@pytest.mark.parametrize(
    "time_sum",
    [
        pytest.param(var("x").time_sum_between(0, param("c")), id="constant"),
        pytest.param(var("x").time_sum_between(0, param("s")), id="scenario-dependent"),
        pytest.param(
            var("x").time_sum_between(0, param("k").eval(0)), id="time-operator"
        ),
        pytest.param(
            var("x").time_sum_between(relative_time(-param("c")), relative_time(0)),
            id="relative-constant",
        ),
        pytest.param(var("x").time_sum_between(0, relative_time(0)), id="mixed"),
    ],
)
def test_time_independent_bounds_are_accepted(time_sum: ExpressionNode) -> None:
    _model(constraint=time_sum)


@pytest.mark.parametrize(
    "time_sum, error",
    [
        pytest.param(
            var("x").time_sum_between(0, param("k")),
            "the time sum bound 'k' depends on time, but must be time-independent. "
            "Declare 'k' with `time-dependent: false`.",
            id="absolute",
        ),
        pytest.param(
            var("x").time_sum_between(relative_time(-param("k")), relative_time(0)),
            "the time sum bound '(t + -(k))' depends on time",
            id="relative-offset",
        ),
        pytest.param(
            var("x").time_sum_between(0, param("k").eval(0) + param("j")),
            "the time sum bound '((k.eval(0.0)) + j)' depends on time, but must be "
            "time-independent. Declare 'j' with `time-dependent: false`.",
            id="names-only-the-time-dependent-parameter",
        ),
        pytest.param(
            var("x").time_sum_between(0, param("k").shift(-1)),
            "the time sum bound '(k.shift(-1.0))' depends on time, but must be "
            "time-independent. Declare 'k' with `time-dependent: false`.",
            id="time-shift-moves-with-t",
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
    ],
)
def test_invalid_bounds_are_rejected_in_constraints(
    time_sum: ExpressionNode, error: str
) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"Model 'm', constraint 'c0': {error}")
    ):
        _model(constraint=time_sum)


def test_invalid_bounds_are_rejected_in_extra_outputs() -> None:
    with pytest.raises(ValueError, match="Model 'm', extra output 'e': "):
        _model(extra_outputs={"e": var("x").time_sum_between(0, param("k"))})


def test_invalid_bounds_are_rejected_in_objective_contributions() -> None:
    with pytest.raises(ValueError, match="Model 'm', objective contribution 'o': "):
        _model(objective=var("x").time_sum_between(0, param("k")).expec())


def test_invalid_bounds_of_a_nested_time_sum_are_rejected() -> None:
    inner = var("x").time_sum_between(0, param("k"))
    with pytest.raises(ValueError, match="the time sum bound 'k' depends on time"):
        _model(constraint=inner.time_sum_between(0, 2))
