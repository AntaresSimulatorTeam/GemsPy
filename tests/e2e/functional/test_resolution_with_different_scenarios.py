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
Tests comparing solving weekly problems scenario-by-scenario vs all scenarios at once.

Study: thermal_heuristic_three_clusters_milp (G1, G2, G3 with availability timeseries).
2 weeks x 2 scenarios. Verifies that the two solving strategies give consistent results.
"""

from pathlib import Path

import linopy
import pytest

from gems.simulation import TimeBlock, build_problem
from gems.study.folder import load_study

xpress_available = pytest.mark.skipif(
    "xpress" not in linopy.available_solvers,
    reason="XPRESS solver not available",
)

STUDIES_DIR = Path(__file__).parent / "studies"
MILP_STUDY_DIR = STUDIES_DIR / "thermal_heuristic_three_clusters_milp"

WEEKS = [range(168), range(168, 336)]
SCENARIOS = [0, 1]

# Expected objective costs [scenario][week]
# With Antares-like XPRESS config (MipRelStop=0.0001): week 1 scenario 0 may stop slightly early
EXPECTED_COST = [
    [78933841, 102109698],
    [17472101, 17424769],
]
# With stricter MipRelStop=0.000001: solver finds the true optimal
EXPECTED_COST_STRICT = [
    [78933742, 102103588],
    [17472101, 17424769],
]

# XPRESS parameters matching Antares solver configuration
_XPRESS_ANTARES_CONFIG = {
    "PRESOLVE": 0,
    "SCALING": 0,
    "FEASTOL": 1e-7,
    "OPTIMALITYTOL": 1e-7,
    "MIPRELSTOP": 0.0001,
}


@xpress_available
def test_one_problem_per_scenario() -> None:
    """Solve each (week, scenario) independently with Antares-like XPRESS config."""
    study = load_study(MILP_STUDY_DIR)
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        for scenario in SCENARIOS:
            problem = build_problem(study, time_block, scenario_ids=[scenario])
            problem.solve(solver_name="xpress", **_XPRESS_ANTARES_CONFIG)
            assert problem.termination_condition == "optimal"
            assert problem.objective_value == pytest.approx(EXPECTED_COST[scenario][week_idx])


@xpress_available
def test_one_problem_per_scenario_with_stricter_mip_gap() -> None:
    """
    Same as above with MipRelStop=0.000001. The stricter gap finds the true optimal
    for scenario 0 week 1 (102103588 instead of 102109698 with the looser gap).
    """
    study = load_study(MILP_STUDY_DIR)
    strict_config = {**_XPRESS_ANTARES_CONFIG, "MIPRELSTOP": 0.000001}
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        for scenario in SCENARIOS:
            problem = build_problem(study, time_block, scenario_ids=[scenario])
            problem.solve(solver_name="xpress", **strict_config)
            assert problem.termination_condition == "optimal"
            assert problem.objective_value == pytest.approx(EXPECTED_COST_STRICT[scenario][week_idx])


@xpress_available
def test_one_problem_for_all_scenarios() -> None:
    """
    Solve all scenarios in a single problem. The objective is expec(cost) = average
    over scenarios. Verify it equals the average of the per-scenario expected costs.
    """
    study = load_study(MILP_STUDY_DIR)
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        problem = build_problem(study, time_block, scenario_ids=SCENARIOS)
        problem.solve(solver_name="xpress", **_XPRESS_ANTARES_CONFIG)
        assert problem.termination_condition == "optimal"
        expected_average = sum(EXPECTED_COST[s][week_idx] for s in SCENARIOS) / len(SCENARIOS)
        assert problem.objective_value == pytest.approx(expected_average)


@xpress_available
def test_one_problem_for_all_scenarios_with_stricter_mip_gap() -> None:
    """
    Same as above with MipRelStop=0.000001. The stricter gap gives the same result
    as solving scenarios independently, confirming consistency between strategies.
    """
    study = load_study(MILP_STUDY_DIR)
    strict_config = {**_XPRESS_ANTARES_CONFIG, "MIPRELSTOP": 0.000001}
    for week_idx, week_range in enumerate(WEEKS):
        time_block = TimeBlock(week_idx + 1, list(week_range))
        problem = build_problem(study, time_block, scenario_ids=SCENARIOS)
        problem.solve(solver_name="xpress", **strict_config)
        assert problem.termination_condition == "optimal"
        expected_average = sum(EXPECTED_COST_STRICT[s][week_idx] for s in SCENARIOS) / len(SCENARIOS)
        assert problem.objective_value == pytest.approx(expected_average)
