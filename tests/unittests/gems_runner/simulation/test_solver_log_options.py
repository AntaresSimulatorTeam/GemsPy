# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for OptimizationProblem.log_output_options (solver-options.logs)."""

import pytest

from gems_runner.simulation.optimization import OptimizationProblem


@pytest.mark.parametrize(
    "solver_name, logs, expected",
    [
        ("highs", True, {"output_flag": True}),
        ("highs", False, {"output_flag": False}),
        ("gurobi", True, {"OutputFlag": 1}),
        ("gurobi", False, {"OutputFlag": 0}),
        ("xpress", True, {"outputlog": 1}),
        ("xpress", False, {"outputlog": 0}),
    ],
)
def test_logs_maps_to_the_solver_native_output_option(
    solver_name: str, logs: bool, expected: dict
) -> None:
    assert OptimizationProblem.log_output_options(solver_name, logs) == expected


def test_unknown_solver_gets_no_output_option() -> None:
    assert OptimizationProblem.log_output_options("some_solver", True) == {}


def test_no_solver_logs_option_is_produced() -> None:
    """'solver_logs' is not a linopy parameter: it would be forwarded to the
    solver as an unknown option."""
    for solver_name in ("highs", "gurobi", "xpress"):
        for logs in (True, False):
            options = OptimizationProblem.log_output_options(solver_name, logs)
            assert "solver_logs" not in options
