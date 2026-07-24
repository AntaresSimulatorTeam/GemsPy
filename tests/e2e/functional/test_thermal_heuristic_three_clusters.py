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
End-to-end test for the thermal three-clusters study using SimulationTable.

Study: tests/e2e/functional/studies/thermal_heuristic_three_clusters_*
Components:
  - N  : node (area model)
  - D  : fixed demand (load model, timeseries demand-ts)
  - G1 : thermal cluster (p_max=410, p_min=180, nb_units=1, timeseries series_G1)
  - G2 : thermal cluster (p_max=90,  p_min=60,  nb_units=3, timeseries series_G2)
  - G3 : thermal cluster (p_max=275, p_min=150, nb_units=4, timeseries series_G3)

2 scenarios x 2 weeks of 168 hours each.
Week 0 uses time indices 0-167, week 1 uses 168-335.
Scenario 0 and scenario 1 correspond to columns 0 and 1 of the timeseries files.
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
from gems_runner.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
)
from tests.e2e.functional.expected_outputs_three_clusters import (
    GEN_G1_ACCURATE,
    GEN_G1_FAST,
    GEN_G1_MILP,
    GEN_G2_ACCURATE,
    GEN_G2_FAST,
    GEN_G2_MILP,
    GEN_G3_ACCURATE,
    GEN_G3_FAST,
    GEN_G3_MILP,
    NODU_G1_ACCURATE,
    NODU_G1_MILP,
    NODU_G2_ACCURATE,
    NODU_G2_MILP,
    NODU_G3_ACCURATE,
    NODU_G3_MILP,
    SPIL_ACCURATE,
    SPIL_FAST,
    SPIL_MILP,
    UNSP_ACCURATE,
    UNSP_FAST,
    UNSP_MILP,
)

STUDIES_DIR = Path(__file__).parent / "studies"

MILP_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_three_clusters_milp"
ACCURATE_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_three_clusters_accurate"
FAST_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_three_clusters_fast"

WEEKS = [range(168), range(168, 336)]
SCENARIOS = [0, 1]

# Expected objective costs [scenario][week]
_COST_MILP = [[78933742, 102103587], [17472101, 17424769]]
_COST_ACCURATE = [[78996726, 102215087 - 69500], [17587733, 17650089]]
_COST_FAST = [
    [79277215 - 630089, 102461792 - 699765],
    [17803738 - 661246, 17720390 - 661246],
]


def _assert_per_timestep(
    st: SimulationTable,
    component_id: str,
    output_id: str,
    expected_values: list,
    time_offset: int = 0,
    abs: float = 1e-6,
) -> None:
    for t, expected in enumerate(expected_values):
        actual = (
            st.component(component_id)
            .output(output_id)
            .value(time_index=time_offset + t, scenario_index=0)
        )
        assert actual == pytest.approx(expected, abs=abs), (  # type: ignore[operator]
            f"{component_id}.{output_id} at t={time_offset + t}: expected {expected}, got {actual}"
        )


def test_milp_version() -> None:
    """Solve weekly problems with MILP (integer commitment variables)."""
    study = load_study(MILP_STUDY_DIR)
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        for scenario in SCENARIOS:
            problem = build_problem(study, time_block, scenario_ids=[scenario])
            problem.solve(solver_name="highs")
            assert problem.termination_condition == "optimal"
            assert problem.objective_value == pytest.approx(
                _COST_MILP[scenario][week_idx]
            )

            st = SimulationTableBuilder().build(problem)
            offset = week_idx * 168
            _assert_per_timestep(
                st, "G1", "generation_power", GEN_G1_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G2", "generation_power", GEN_G2_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G3", "generation_power", GEN_G3_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G1", "num_units_on", NODU_G1_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G2", "num_units_on", NODU_G2_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G3", "num_units_on", NODU_G3_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "N", "spilled_energy", SPIL_MILP[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "N", "unsupplied_energy", UNSP_MILP[scenario][week_idx], offset
            )


def test_accurate_heuristic() -> None:
    """
    Solve weekly problems with the accurate heuristic of Antares.
    The accurate heuristic rounds up the number of on units from the LP relaxation.
    """
    study = load_study(ACCURATE_STUDY_DIR)
    optim_config = load_optim_config(ACCURATE_STUDY_DIR / "input" / "optim-config.yml")
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        for scenario in SCENARIOS:
            problem = build_problem(study, time_block, scenario_ids=[scenario])
            problem.solve(solver_name="highs")
            if should_apply_heuristics(study):
                apply_thermal_heuristics(problem, optim_config, [scenario])
                problem.solve(solver_name="highs")
            assert problem.termination_condition == "optimal"
            assert problem.objective_value == pytest.approx(
                _COST_ACCURATE[scenario][week_idx]
            )

            st = SimulationTableBuilder().build(problem)
            offset = week_idx * 168
            _assert_per_timestep(
                st,
                "G1",
                "generation_power",
                GEN_G1_ACCURATE[scenario][week_idx],
                offset,
            )
            _assert_per_timestep(
                st,
                "G2",
                "generation_power",
                GEN_G2_ACCURATE[scenario][week_idx],
                offset,
            )
            _assert_per_timestep(
                st,
                "G3",
                "generation_power",
                GEN_G3_ACCURATE[scenario][week_idx],
                offset,
            )
            _assert_per_timestep(
                st, "G1", "num_units_on", NODU_G1_ACCURATE[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G2", "num_units_on", NODU_G2_ACCURATE[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G3", "num_units_on", NODU_G3_ACCURATE[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "N", "spilled_energy", SPIL_ACCURATE[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "N", "unsupplied_energy", UNSP_ACCURATE[scenario][week_idx], offset
            )


def test_fast_heuristic() -> None:
    """
    Solve weekly problems with the fast heuristic of Antares.
    The fast heuristic uses slot-based scheduling of commitment decisions.
    """
    study = load_study(FAST_STUDY_DIR)
    optim_config = load_optim_config(FAST_STUDY_DIR / "input" / "optim-config.yml")
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        for scenario in SCENARIOS:
            problem = build_problem(study, time_block, scenario_ids=[scenario])
            problem.solve(solver_name="highs")
            if should_apply_heuristics(study):
                apply_thermal_heuristics(problem, optim_config, [scenario])
                problem.solve(solver_name="highs")
            assert problem.termination_condition == "optimal"
            assert problem.objective_value == pytest.approx(
                _COST_FAST[scenario][week_idx]
            )

            st = SimulationTableBuilder().build(problem)
            offset = week_idx * 168
            _assert_per_timestep(
                st, "G1", "generation_power", GEN_G1_FAST[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G2", "generation_power", GEN_G2_FAST[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "G3", "generation_power", GEN_G3_FAST[scenario][week_idx], offset
            )
            # fast heuristic does not check num_units_on (slot-based, not per-unit)
            _assert_per_timestep(
                st, "N", "spilled_energy", SPIL_FAST[scenario][week_idx], offset
            )
            _assert_per_timestep(
                st, "N", "unsupplied_energy", UNSP_FAST[scenario][week_idx], offset
            )
