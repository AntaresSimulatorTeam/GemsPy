# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

Study: tests/e2e/functional/studies/thermal_heuristic_three_clusters
Components:
  - N  : node (area model)
  - D  : fixed demand (load model, timeseries demand-ts)
  - G1 : thermal cluster (p_max=410, p_min=180, num_units=1, timeseries series_G1)
  - G2 : thermal cluster (p_max=90,  p_min=60,  num_units=3, timeseries series_G2)
  - G3 : thermal cluster (p_max=275, p_min=150, num_units=4, timeseries series_G3)

2 scenarios x 2 weeks of 168 hours each.
Week 0 uses time indices 0-167, week 1 uses 168-335.
Scenario 0 and scenario 1 correspond to columns 0 and 1 of the timeseries files.
"""

import pytest

from gems_craft.optim_config.parsing import ResolutionConfig, ResolutionMode
from gems_runner.session.session import SimulationSession
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
    NODU_G1_FAST,
    NODU_G1_MILP,
    NODU_G2_ACCURATE,
    NODU_G2_FAST,
    NODU_G2_MILP,
    NODU_G3_ACCURATE,
    NODU_G3_FAST,
    NODU_G3_MILP,
    SPIL_ACCURATE,
    SPIL_FAST,
    SPIL_MILP,
    UNSP_ACCURATE,
    UNSP_FAST,
    UNSP_MILP,
)
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_for,
    total_output_sum,
)

CASE_ID = "three_clusters"

WEEKS = [range(168), range(168, 336)]
SCENARIOS = [0, 1]
THERMAL_COMPONENTS = ["G1", "G2", "G3"]

# Expected objective costs [scenario][week]
_COST_MILP = [[78933742, 102103587], [17472101, 17424769]]
_COST_ACCURATE = [[78996726, 102145587], [17587733, 17650089]]
_COST_FAST = [
    [78647126, 101762027],
    [17142492, 17059144],
]  # non_prop_cost not included in objective for fast models

# Expected total non_prop_cost (G1+G2+G3) [scenario][week]
_NON_PROP_COST_MILP = [[470566, 630191], [407736, 526227]]
_NON_PROP_COST_ACCURATE = [[560556, 560721], [313783, 313787]]
_NON_PROP_COST_FAST = [[699554, 838690], [1008660, 1008670]]


def test_milp_version() -> None:
    """Solve weekly problems with MILP (integer commitment variables)."""
    study = build_thermal_study(CASE_ID, "milp")
    config = optim_config_for(
        CASE_ID,
        "milp",
        WEEKS[0].start,
        WEEKS[-1].stop - 1,
        SCENARIOS,
        resolution=ResolutionConfig(
            mode=ResolutionMode.PARALLEL_SUBPROBLEMS, block_length=168
        ),
    )
    st = SimulationSession(study, config).run()
    # objective-value rows carry no scenario_index; recover [scenario][week] from
    # _run_parallel's solve order (scenarios outer in SCENARIOS order, weeks inner).
    objective_values = st.data.loc[
        st.data["output"] == "objective-value", "value"
    ].tolist()
    for scenario in SCENARIOS:
        for week_idx in range(len(WEEKS)):
            objective = objective_values[scenario * len(WEEKS) + week_idx]
            assert objective == pytest.approx(_COST_MILP[scenario][week_idx])

            assert total_output_sum(
                st,
                THERMAL_COMPONENTS,
                "non_prop_cost",
                scenario_index=scenario,
                time_range=WEEKS[week_idx],
            ) == pytest.approx(_NON_PROP_COST_MILP[scenario][week_idx])

        check_output(
            st,
            "G1",
            "generation_power",
            GEN_G1_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "generation_power",
            GEN_G2_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "generation_power",
            GEN_G3_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G1",
            "num_units_on",
            NODU_G1_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "num_units_on",
            NODU_G2_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "num_units_on",
            NODU_G3_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "spilled_energy",
            SPIL_MILP[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "unsupplied_energy",
            UNSP_MILP[scenario],
            scenario_index=scenario,
        )


def test_accurate_heuristic() -> None:
    """
    Solve weekly problems with the accurate heuristic.
    The accurate heuristic rounds up the number of on units from the LP relaxation.
    """
    study = build_thermal_study(CASE_ID, "accurate")
    config = optim_config_for(
        CASE_ID,
        "accurate",
        WEEKS[0].start,
        WEEKS[-1].stop - 1,
        SCENARIOS,
        resolution=ResolutionConfig(
            mode=ResolutionMode.PARALLEL_SUBPROBLEMS, block_length=168
        ),
    )
    st = SimulationSession(study, config).run()
    # objective-value rows carry no scenario_index; recover [scenario][week] from
    # _run_parallel's solve order (scenarios outer in SCENARIOS order, weeks inner).
    objective_values = st.data.loc[
        st.data["output"] == "objective-value", "value"
    ].tolist()
    for scenario in SCENARIOS:
        for week_idx in range(len(WEEKS)):
            objective = objective_values[scenario * len(WEEKS) + week_idx]
            assert objective == pytest.approx(_COST_ACCURATE[scenario][week_idx])

            assert total_output_sum(
                st,
                THERMAL_COMPONENTS,
                "non_prop_cost",
                scenario_index=scenario,
                time_range=WEEKS[week_idx],
            ) == pytest.approx(_NON_PROP_COST_ACCURATE[scenario][week_idx])

        check_output(
            st,
            "G1",
            "generation_power",
            GEN_G1_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "generation_power",
            GEN_G2_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "generation_power",
            GEN_G3_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G1",
            "num_units_on",
            NODU_G1_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "num_units_on",
            NODU_G2_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "num_units_on",
            NODU_G3_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "spilled_energy",
            SPIL_ACCURATE[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "unsupplied_energy",
            UNSP_ACCURATE[scenario],
            scenario_index=scenario,
        )


def test_fast_heuristic() -> None:
    """
    Solve weekly problems with the fast heuristic.
    The fast heuristic uses slot-based scheduling of commitment decisions.
    """
    study = build_thermal_study(CASE_ID, "fast")
    config = optim_config_for(
        CASE_ID,
        "fast",
        WEEKS[0].start,
        WEEKS[-1].stop - 1,
        SCENARIOS,
        resolution=ResolutionConfig(
            mode=ResolutionMode.PARALLEL_SUBPROBLEMS, block_length=168
        ),
    )
    st = SimulationSession(study, config).run()
    # objective-value rows carry no scenario_index; recover [scenario][week] from
    # _run_parallel's solve order (scenarios outer in SCENARIOS order, weeks inner).
    objective_values = st.data.loc[
        st.data["output"] == "objective-value", "value"
    ].tolist()

    for scenario in SCENARIOS:
        for week_idx in range(len(WEEKS)):
            objective = objective_values[scenario * len(WEEKS) + week_idx]
            assert objective == pytest.approx(_COST_FAST[scenario][week_idx])

            assert total_output_sum(
                st,
                THERMAL_COMPONENTS,
                "non_prop_cost",
                scenario_index=scenario,
                time_range=WEEKS[week_idx],
            ) == pytest.approx(_NON_PROP_COST_FAST[scenario][week_idx])

        check_output(
            st,
            "G1",
            "generation_power",
            GEN_G1_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "generation_power",
            GEN_G2_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "generation_power",
            GEN_G3_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G1",
            "num_units_on",
            NODU_G1_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G2",
            "num_units_on",
            NODU_G2_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "G3",
            "num_units_on",
            NODU_G3_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "spilled_energy",
            SPIL_FAST[scenario],
            scenario_index=scenario,
        )
        check_output(
            st,
            "N",
            "unsupplied_energy",
            UNSP_FAST[scenario],
            scenario_index=scenario,
        )
