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
End-to-end tests for the thermal one-cluster-with-ramp study.

Study: tests/e2e/functional/studies/thermal_heuristic_one_cluster_with_ramp_*
Components:
  - N : node (area model, spillage_cost=100, unsupplied_energy_cost=10000)
  - D : fixed demand (ramps 0->300 MW over 7 hours, then back to 0 at t=12, zeros thereafter)
  - G : thermal cluster (p_max=500, p_min=100, nb_units_max=3, market_bid_cost=10, startup_cost=10,
                         d_min_up=1, d_min_down=1) with ramping constraints (ramp_up=30, ramp_down=30)

The study is solved over a full week (168 timesteps) but only the first 12-13 timesteps
are asserted, since that is the window where the demand ramp (and therefore the
interesting heuristic behavior) takes place; all remaining timesteps are flat at zero.

Three variants of the same study are compared:
  - test_milp_version: reference solution with the full MILP (integer nb_units_on/
    starting/stopping), used as the ground truth for objective value and dispatch.
  - test_accurate_heuristic: the accurate heuristic's second LP has no integer
    constraint on the number of starting/stopping units, so nb_units_on ends up
    fractional even though the relaxed LP is otherwise feasible.
  - test_fast_heuristic: the fast heuristic rounds nb_units_on to integers (NODU=1
    over the ramp window), but a single unit ramping 100 MW in one step exceeds the
    30 MW/unit ramp constraint, so the resulting dispatch is ramp-infeasible even
    though the solver reports an optimal LP solution.
"""

from pathlib import Path

import pytest

from gems.optim_config.parsing import load_optim_config
from gems.simulation import TimeBlock, build_problem
from gems.simulation.heuristic_runner import (
    apply_thermal_heuristics,
    should_apply_heuristics,
)
from gems.simulation.simulation_table import SimulationTable, SimulationTableBuilder
from gems.study.folder import load_study

STUDIES_DIR = Path(__file__).parent / "studies"
MILP_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_with_ramp_milp"
ACCURATE_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_with_ramp_accurate"
FAST_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_with_ramp_fast"


def _assert_per_timestep(
    st: SimulationTable,
    component_id: str,
    output_id: str,
    expected_values: list,
    abs: float = 1e-2,
) -> None:
    for t, expected in enumerate(expected_values):
        actual = (
            st.component(component_id)
            .output(output_id)
            .value(time_index=t, scenario_index=0)
        )
        assert actual == pytest.approx(expected, abs=abs), (  # type: ignore[operator]
            f"{component_id}.{output_id} at t={t}: expected {expected}, got {actual}"
        )


def test_milp_version() -> None:
    """Solve weekly problem with one cluster and ramp constraints with milp."""
    study = load_study(MILP_STUDY_DIR)
    time_block = TimeBlock(1, list(range(168)))
    problem = build_problem(study, time_block, scenario_ids=[0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(29040)

    st = SimulationTableBuilder().build(problem)
    _assert_per_timestep(
        st,
        "G",
        "generation_power",
        [
            0.0,
            100.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            250.0,
            200.0,
            150.0,
            100.0,
            100.0,
        ],
    )
    _assert_per_timestep(
        st,
        "G",
        "num_units_on",
        [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
    )
    _assert_per_timestep(st, "N", "spilled_energy", [0.0, 50.0] + [0.0] * 9 + [50.0])
    _assert_per_timestep(st, "N", "unsupplied_energy", [0.0] * 12)


def test_accurate_heuristic() -> None:
    """
    With the ramp model, the accurate heuristic's second LP produces fractional
    num_units_on values, because it has no integer constraint on the number of
    starting/stopping units.
    """
    study = load_study(ACCURATE_STUDY_DIR)
    time_block = TimeBlock(1, list(range(168)))
    optim_config = load_optim_config(ACCURATE_STUDY_DIR / "input" / "optim-config.yml")
    problem = build_problem(study, time_block, scenario_ids=[0])
    problem.solve(solver_name="highs")
    if should_apply_heuristics(study):
        apply_thermal_heuristics(problem, optim_config, [0])
        problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(29011.4616736)

    st = SimulationTableBuilder().build(problem)
    _assert_per_timestep(
        st,
        "G",
        "num_units_on",
        [
            0.0,
            1.0,
            1.0,
            1.04,
            1.07,
            1.11,
            1.14,
            1.11,
            1.08,
            1.04,
            1.0,
            1.0,
            0.0,
        ],
    )  # non-integer values, as expected from the accurate heuristic's relaxed LP
    _assert_per_timestep(
        st,
        "G",
        "num_units_starting",
        [
            0.0,
            1.0,
            0.0,
            0.04,
            0.04,
            0.03,
            0.03,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    _assert_per_timestep(
        st,
        "G",
        "num_units_stopping",
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03,
            0.04,
            0.04,
            0.04,
            0.0,
            1.0,
        ],
    )


def test_fast_heuristic() -> None:
    """
    With the ramp model, the fast heuristic rounds num_units_on to 1 over the demand
    ramp, but a single unit ramping 100 MW in one step violates the ramp constraint of
    30 MW/unit: the resulting dispatch is ramp-infeasible even though the LP reports
    an optimal solution.
    """
    study = load_study(FAST_STUDY_DIR)
    time_block = TimeBlock(1, list(range(168)))
    optim_config = load_optim_config(FAST_STUDY_DIR / "input" / "optim-config.yml")
    problem = build_problem(study, time_block, scenario_ids=[0])
    problem.solve(solver_name="highs")
    if should_apply_heuristics(study):
        apply_thermal_heuristics(problem, optim_config, [0])
        problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(29000)

    st = SimulationTableBuilder().build(problem)
    _assert_per_timestep(
        st,
        "G",
        "generation_power",
        [
            0.0,
            100.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            250.0,
            200.0,
            150.0,
            100.0,
            100.0,
        ],
    )
    _assert_per_timestep(st, "G", "num_units_on", [0.0] + [1.0] * 11)
