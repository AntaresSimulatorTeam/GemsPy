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

import pytest

from gems.simulation import TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.study.folder import load_study

STUDIES_DIR = Path(__file__).parent / "studies"


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
            },
        ),
        (
            "10_5_1",
            {
                "objective": 1500.0,
                "base_zone.price": 50.0,
                "gas_base_zone.generation_reduced_cost": -40.0,
                "oil_base_zone.generation_reduced_cost": 0.0,
            },
        ),
    ],
)
def test_dual_reduced_cost_single_timestep(study_id: str, expected: dict) -> None:
    """Verify nodal price (dual) and reduced costs for single-timestep studies."""
    study = load_study(STUDIES_DIR / study_id)
    time_block = TimeBlock(1, [0])
    problem = build_problem(study, time_block, [0])
    problem.solve(solver_name="highs")

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


def test_dual_reduced_cost_multi_timestep() -> None:
    """Verify nodal prices and reduced costs for a 3-timestep study (10_5_2)."""
    study = load_study(STUDIES_DIR / "10_5_2")
    time_block = TimeBlock(1, [0, 1, 2])
    problem = build_problem(study, time_block, [0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(27550.0)

    st = SimulationTableBuilder().build(problem)

    # Nodal prices at t=0,1,2
    for t, expected_price in enumerate([10.0, 15.0, 20000.0]):
        price = (
            st.component("base_zone")
            .output("price")
            .value(time_index=t, scenario_index=0)
        )
        assert price == pytest.approx(
            expected_price
        ), f"price at t={t}: expected {expected_price}, got {price}"

    # Gas generator reduced costs at t=0,1,2
    for t, expected_rc in enumerate([0.0, 0.0, -19960.0]):
        rc = (
            st.component("gas_base_zone")
            .output("generation_reduced_cost")
            .value(time_index=t, scenario_index=0)
        )
        assert rc == pytest.approx(
            expected_rc, abs=1e-3
        ), f"gas RC at t={t}: expected {expected_rc}, got {rc}"

    # Oil generator reduced costs at t=0,1,2
    for t, expected_rc in enumerate([20.0, -5.0, -19990.0]):
        rc = (
            st.component("oil_base_zone")
            .output("generation_reduced_cost")
            .value(time_index=t, scenario_index=0)
        )
        assert rc == pytest.approx(
            expected_rc, abs=1e-3
        ), f"oil RC at t={t}: expected {expected_rc}, got {rc}"


def test_dual_in_constraint_is_rejected() -> None:
    """dual() in a constraint expression must be caught by the library resolver."""
    from gems.expression.parsing.parse_expression import (
        ModelIdentifiers,
        parse_expression,
    )
    from gems.model.resolve_library import _forbid_dual_or_rc

    ids = ModelIdentifiers(
        variables={"x"},
        parameters=set(),
        constraints={"balance"},
    )
    expr = parse_expression("dual(balance) + x", ids)
    with pytest.raises(ValueError, match="Operators dual/reduced_cost are not allowed"):
        _forbid_dual_or_rc(expr, "constraint 'bad'")


def test_reduced_cost_in_objective_is_rejected() -> None:
    """reduced_cost() in an objective contribution must be caught by the library resolver."""
    from gems.expression.parsing.parse_expression import (
        ModelIdentifiers,
        parse_expression,
    )
    from gems.model.resolve_library import _forbid_dual_or_rc

    ids = ModelIdentifiers(
        variables={"x"},
        parameters=set(),
        constraints=set(),
    )
    expr = parse_expression("reduced_cost(x)", ids)
    with pytest.raises(ValueError, match="Operators dual/reduced_cost are not allowed"):
        _forbid_dual_or_rc(expr, "objective contribution 'obj'")
