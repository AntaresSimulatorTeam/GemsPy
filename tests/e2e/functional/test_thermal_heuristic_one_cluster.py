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
End-to-end test for the thermal one-cluster study using SimulationTable.

Study: tests/e2e/functional/data/thermal_heuristic_one_cluster
Components:
  - N  : node (NODE_BALANCE_MODEL)
  - D  : fixed demand (FIXED_DEMAND, timeseries demand-ts)
  - G  : thermal generator (GEN, p_max=1000, p_min=700, cost=50, ...)
  - S  : spillage (SPI, cost=0)
  - U  : unsupplied energy (UNSP, cost=1000)

The study directory must have the load_study-compatible structure:
  input/system.yml
  input/model-libraries/<library>.yml
  input/data-series/demand-ts.txt   (copy of data/thermal_heuristic_one_cluster/demand-ts.txt)
"""

from pathlib import Path

import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.heuristic_runner import (
    apply_thermal_heuristics,
    should_apply_heuristics,
)
from gems_runner.simulation.simulation_table import SimulationTableBuilder

STUDIES_DIR = Path(__file__).parent / "studies"

MILP_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_milp"
LP_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_lp"
ACCURATE_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_accurate"
FAST_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_one_cluster_fast"


def test_milp_version() -> None:
    """
    Model on 168 time steps with one thermal generation and demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal milp solution consists in turning on two thermal plants at the begining, turning on a third thermal plant at the 13th hour and turning off the first thermal plant at the 14th hour, the other two thermal plants stay on for the rest of the week producing 1000MW each. At the 13th hour, the production is [700,700,700] to satisfy Pmin constraints.

    The optimal cost is then :
          50 x 2 x 1000 x 167 (prod step 1-12 and 14-168)
        + 50 x 3 x 700 (prod step 13)
        + 50 (start up step 13)
        + 2 x 1 x 167 (fixed cost step 1-12 and 14-168)
        + 3 x 1 (fixed cost step 13)
        = 16 805 387
    """
    study = load_study(MILP_STUDY_DIR)
    time_block = TimeBlock(1, list(range(168)))
    problem = build_problem(study, time_block, scenario_ids=[0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(16805387)

    st = SimulationTableBuilder().build(problem)

    def assert_output_per_timestep(
        component_id: str,
        output_id: str,
        expected_values: list,
        abs: float = 1e-6,
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

    assert_output_per_timestep(
        "G", "generation_power", [2000 if t != 12 else 2100 for t in range(168)]
    )
    assert_output_per_timestep(
        "G", "num_units_on", [2 if t != 12 else 3 for t in range(168)]
    )
    assert_output_per_timestep("N", "unsupplied_energy", [0.0] * 168)
    assert_output_per_timestep(
        "N", "spilled_energy", [0 if t != 12 else 50 for t in range(168)]
    )


def test_lp_version() -> None:
    """
    Model on 168 time steps with one thermal generation and one demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal solution of the linear relaxation consists in producing exactly the demand at each hour. The number of on units is equal to the production divided by P_max.

    The optimal cost is then :
          50 x 2000 x 167 (prod step 1-12 and 14-168)
        + 50 x 2050 (prod step 13)
        + 2 x 1 x 168 (fixed cost step 1-12 and 14-168)
        + 2050/1000 x 1 (fixed cost step 13)
        + 0,05 x 50 (start up cost step 13)
        = 16 802 840,55
    """
    study = load_study(LP_STUDY_DIR)
    time_block = TimeBlock(1, list(range(168)))
    problem = build_problem(study, time_block, scenario_ids=[0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(16802840.55)

    st = SimulationTableBuilder().build(problem)

    def assert_output_per_timestep(
        component_id: str,
        output_id: str,
        expected_values: list,
        abs: float = 1e-6,
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

    assert_output_per_timestep(
        "G", "generation_power", [2000 if t != 12 else 2050 for t in range(168)]
    )
    assert_output_per_timestep(
        "G", "num_units_on", [2 if t != 12 else 2.05 for t in range(168)]
    )
    assert_output_per_timestep("N", "unsupplied_energy", [0.0] * 168)
    assert_output_per_timestep("N", "spilled_energy", [0.0] * 168)


def test_accurate_heuristic() -> None:
    """
    Solve the same problem as before with the heuristic accurate of Antares. The accurate heuristic is able to retrieve the milp optimal solution because when the number of on units found in the linear relaxation is ceiled, we found the optimal number of on units which is already feasible.
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
    assert problem.objective_value == pytest.approx(16805387)

    st = SimulationTableBuilder().build(problem)

    def assert_output_per_timestep(
        component_id: str,
        output_id: str,
        expected_values: list,
        abs: float = 1e-6,
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

    assert_output_per_timestep(
        "G", "generation_power", [2000 if t != 12 else 2100 for t in range(168)]
    )
    assert_output_per_timestep(
        "G", "num_units_on", [2 if t != 12 else 3 for t in range(168)]
    )
    assert_output_per_timestep("N", "unsupplied_energy", [0.0] * 168)
    assert_output_per_timestep(
        "N", "spilled_energy", [0 if t != 12 else 50 for t in range(168)]
    )


def test_fast_heuristic() -> None:
    """
    Solve the same problem as before with the heuristic fast of Antares
    Model on 168 time steps with one thermal generation and one demand on a single node.
        - Demand is constant to 2000 MW except for the 13th hour for which it is 2050 MW
        - Thermal generation is characterized with:
            - P_min = 700 MW
            - P_max = 1000 MW
            - Min up time = 3
            - Min down time = 10
            - Generation cost = 50€ / MWh
            - Startup cost = 50
            - Fixed cost = 1 /h
            - Number of unit = 3
        - Unsupplied energy = 1000 €/MWh
        - Spillage = 0 €/MWh

    The optimal solution consists in having 3 units turned on between time steps 10 and 19 with production equal to 2100 to respect pmin and 2 the rest of the time. Fast heuristic turns on 3 units for 10 timesteps because min down time is equal to 10.

    The optimal cost is then :
          50 x 2000 x 158 (prod step 1-9 and 20-168)
        + 50 x 2100 x 10 (prod step 10-19)
        = 16 850 000
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
    assert problem.objective_value == pytest.approx(16850000)

    st = SimulationTableBuilder().build(problem)

    def assert_output_per_timestep(
        component_id: str,
        output_id: str,
        expected_values: list,
        abs: float = 1e-6,
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

    assert_output_per_timestep(
        "G",
        "generation_power",
        [2000 if t not in [i for i in range(10, 20)] else 2100 for t in range(168)],
    )
    assert_output_per_timestep(
        "G",
        "num_units_on",
        [2 if t not in [i for i in range(10, 20)] else 3 for t in range(168)],
    )
    assert_output_per_timestep("N", "unsupplied_energy", [0.0] * 168)
    assert_output_per_timestep(
        "N",
        "spilled_energy",
        [
            0 if t not in [i for i in range(10, 20)] else (50 if t == 12 else 100)
            for t in range(168)
        ],
    )
