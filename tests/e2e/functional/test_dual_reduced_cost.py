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

"""
End-to-end tests for the dual() and reduced_cost() extra-output operators.

Studies 10_5 and 10_5_1: single timestep, two generators, one load.
Study 10_5_2: three timesteps, time-varying costs and loads.

Negative tests verify that dual/reduced_cost are rejected when used in
contexts where they are not allowed (constraints, objectives).
"""

from pathlib import Path
from typing import List

import pytest

from gems.simulation import TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.study.folder import load_study

STUDIES_DIR = Path(__file__).parent / "studies"


def _available_solvers() -> List[str]:
    solvers = ["highs"]
    try:
        import xpress  # noqa: F401

        solvers.append("xpress")
    except ImportError:
        pass
    try:
        import gurobipy  # noqa: F401

        solvers.append("gurobi")
    except ImportError:
        pass
    return solvers


_SOLVER_PARAMS = [
    pytest.param(
        s,
        marks=pytest.mark.skipif(
            s not in _available_solvers(), reason=f"{s} not installed"
        ),
    )
    for s in ["highs", "xpress", "gurobi"]
]


# Reduced cost implementation is solver specific, tests are parametrized by solver to test each implementation
@pytest.mark.parametrize(
    "study_id, expected",
    [
        (
            "10_5",
            {
                "objective": 900.0,
                "base_zone.price": 10.0,
                "gas_base_zone.generation_reduced_cost": 0.0,
                "oil_base_zone.generation_reduced_cost": 40.0,
                "gas_base_zone.profit": 0,
                "oil_base_zone.profit": 0,
            },
        ),
        (
            "10_5_1",
            {
                "objective": 1500.0,
                "base_zone.price": 50.0,
                "gas_base_zone.generation_reduced_cost": -40.0,
                "oil_base_zone.generation_reduced_cost": 0.0,
                "gas_base_zone.profit": 4000,
                "oil_base_zone.profit": 0,
            },
        ),
    ],
)
@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_dual_reduced_cost_single_timestep(
    study_id: str, expected: dict, solver_name: str
) -> None:
    """Verify nodal price (dual) and reduced costs for single-timestep studies."""
    study = load_study(STUDIES_DIR / study_id)
    time_block = TimeBlock(1, [0])
    problem = build_problem(study, time_block, [0])
    problem.solve(solver_name=solver_name)

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(expected["objective"])

    st = SimulationTableBuilder().build(problem)

    price = (
        st.component("base_zone").output("price").value(time_index=0, scenario_index=0)
    )
    assert price == pytest.approx(expected["base_zone.price"])

    gas_rc = (
        st.component("gas_base_zone")
        .output("generation_reduced_cost")
        .value(time_index=0, scenario_index=0)
    )
    assert gas_rc == pytest.approx(expected["gas_base_zone.generation_reduced_cost"])

    oil_rc = (
        st.component("oil_base_zone")
        .output("generation_reduced_cost")
        .value(time_index=0, scenario_index=0)
    )
    assert oil_rc == pytest.approx(expected["oil_base_zone.generation_reduced_cost"])

    gas_profit = (
        st.component("gas_base_zone")
        .output("profit")
        .value(time_index=0, scenario_index=0)
    )
    assert gas_profit == pytest.approx(expected["gas_base_zone.profit"])

    oil_profit = (
        st.component("oil_base_zone")
        .output("profit")
        .value(time_index=0, scenario_index=0)
    )
    assert oil_profit == pytest.approx(expected["oil_base_zone.profit"])


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_dual_reduced_cost_multi_timestep(solver_name: str) -> None:
    """Verify nodal prices and reduced costs for a 3-timestep study (10_5_2)."""
    study = load_study(STUDIES_DIR / "10_5_2")
    time_block = TimeBlock(1, [0, 1, 2])
    problem = build_problem(study, time_block, [0])
    problem.solve(solver_name=solver_name)

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(27550.0)

    st = SimulationTableBuilder().build(problem)

    def assert_output_per_timestep(
        component_id: str, output_id: str, expected_values: list, abs: float = 1e-9
    ) -> None:
        for t, expected in enumerate(expected_values):
            actual = (
                st.component(component_id)
                .output(output_id)
                .value(time_index=t, scenario_index=0)
            )
            assert actual == pytest.approx(
                expected, abs=abs
            ), f"{component_id}.{output_id} at t={t}: expected {expected}, got {actual}"

    assert_output_per_timestep("base_zone", "price", [10.0, 15.0, 20000.0])
    assert_output_per_timestep(
        "gas_base_zone", "generation_reduced_cost", [0.0, 0.0, -19960.0], abs=1e-3
    )
    assert_output_per_timestep(
        "oil_base_zone", "generation_reduced_cost", [20.0, -5.0, -19990.0], abs=1e-3
    )
    assert_output_per_timestep(
        "gas_base_zone", "profit", [0.0, 0.0, 1996000.0], abs=1e-3
    )
    assert_output_per_timestep(
        "oil_base_zone", "profit", [0.0, 500.0, 1999000.0], abs=1e-3
    )
