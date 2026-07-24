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
End-to-end test for the thermal two-clusters-with-binding-constraint study.

Study: tests/e2e/functional/studies/thermal_heuristic_two_clusters_with_bc
Components:
  - N               : node (area model, spillage_cost=0, unsupplied_cost=1000)
  - D               : fixed demand (2000 MW except t=12 where demand=2050 MW)
  - G1              : thermal cluster (p_max=1000, p_min=700, nb_units=3, cost=50)
  - G2              : thermal cluster (p_max=500,  p_min=50,  nb_units=2, cost=150)
  - upper_bound_sum : binding constraint (G1 + G2 <= 2050 at all times)

Accurate and fast heuristics are INFEASIBLE because the heuristic forces G1 to run
more units than the binding constraint allows.
"""

import pytest

from gems_runner.session.session import SimulationSession
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_for,
)

CASE_ID = "two_clusters_with_bc"


def test_milp_version() -> None:
    """
    Solve with MILP. Optimal solution: G1 always on (2 units, 2000 MW); G2 turns on at t=12
    (demand=2050), stays on 4 timesteps due to min_up=4, producing at p_min=50 MW.
    Spillage at t=13-15 is free (spillage_cost=0).
    """
    study = build_thermal_study(CASE_ID, "milp")
    config = optim_config_for(CASE_ID, "milp", 0, 167, [0])
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(16822864)

    # At t=13-15: G2 still on (min_up=4), G1 reduces to 1950 so total = 1950+50 = 2000 = demand (no spillage)
    check_output(
        st,
        "G1",
        "generation_power",
        [1950.0 if t in range(13, 16) else 2000.0 for t in range(168)],
    )
    check_output(
        st,
        "G2",
        "generation_power",
        [50.0 if t in range(12, 16) else 0.0 for t in range(168)],
    )
    check_output(st, "G1", "num_units_on", [2.0] * 168)
    check_output(
        st,
        "G2",
        "num_units_on",
        [1.0 if t in range(12, 16) else 0.0 for t in range(168)],
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_lp_version() -> None:
    """
    Solve with LP relaxation. G1 covers all demand (fractional units), G2 is never needed
    because G1 alone can meet the binding constraint at t=12 (2050 <= 2050).
    """
    study = build_thermal_study(CASE_ID, "lp")
    config = optim_config_for(CASE_ID, "lp", 0, 167, [0])
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(16802840.55)

    check_output(
        st,
        "G1",
        "generation_power",
        [2000.0 if t != 12 else 2050.0 for t in range(168)],
    )
    check_output(st, "G2", "generation_power", [0.0] * 168)
    check_output(
        st, "G1", "num_units_on", [2.0 if t != 12 else 2.05 for t in range(168)]
    )
    check_output(st, "G2", "num_units_on", [0.0] * 168)
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_accurate_heuristic() -> None:
    """
    Solve with the accurate heuristic. The heuristic rounds G1 num_units_on up:
    at t=12, LP gives 2.05 units -> ceil = 3, which forces G1 >= 3*700 = 2100 MW.
    Combined with the binding constraint G1+G2 <= 2050, the problem is INFEASIBLE.
    """
    study = build_thermal_study(CASE_ID, "accurate")
    config = optim_config_for(CASE_ID, "accurate", 0, 167, [0])

    with pytest.raises(RuntimeError, match="infeasible"):
        SimulationSession(study, config).run()


def test_fast_heuristic() -> None:
    """
    Solve with the fast heuristic. The heuristic sets G1 min_generating to 3*700=2100 MW
    for timesteps t=10-19 (slot including t=12). This violates G1+G2 <= 2050 -> INFEASIBLE.
    """
    study = build_thermal_study(CASE_ID, "fast")
    config = optim_config_for(CASE_ID, "fast", 0, 167, [0])

    with pytest.raises(RuntimeError, match="infeasible"):
        SimulationSession(study, config).run()
