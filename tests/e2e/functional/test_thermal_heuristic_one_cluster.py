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
End-to-end test for the thermal one-cluster study using SimulationTable.

Study: tests/e2e/functional/studies/thermal_heuristic_one_cluster
Model on 168 time steps with one thermal generation and one demand on a single node.
Components:
  - N : node (NODE_BALANCE_MODEL)
  - D : fixed demand (FIXED_DEMAND, timeseries demand-ts). Demand is constant to
        2000 MW except for the 13th hour for which it is 2050 MW
  - G : thermal generator (GEN):
        - P_min = 700 MW, P_max = 1000 MW
        - Min up time = 3, min down time = 10
        - Generation cost = 50€ / MWh, startup cost = 50, fixed cost = 1 /h
        - Number of units = 3
  - S : spillage (SPI, cost = 0 €/MWh)
  - U : unsupplied energy (UNSP, cost = 1000 €/MWh)
"""

import pytest

from gems_runner.session.session import SimulationSession
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_for,
)

CASE_ID = "one_cluster"


def test_milp_version() -> None:
    """
    The optimal milp solution consists in turning on two thermal plants at the begining,
    turning on a third thermal plant at the 13th hour and turning off the first thermal
    plant at the 14th hour, the other two thermal plants stay on for the rest of the
    week producing 1000MW each. At the 13th hour, the production is [700,700,700] to
    satisfy Pmin constraints. This induces 50 MW of spillage at the 13th hour, but this
    costs 0 since spillage is free.

    The optimal cost is then :
          50 x 2 x 1000 x 167 (prod step 1-12 and 14-168)
        + 50 x 3 x 700 (prod step 13)
        + 50 (start up step 13)
        + 2 x 1 x 167 (fixed cost step 1-12 and 14-168)
        + 3 x 1 (fixed cost step 13)
        = 16 805 387
    """
    study = build_thermal_study(CASE_ID, "milp")
    config = optim_config_for(CASE_ID, "milp", 0, 167, [0])
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(16805387)
    check_output(
        st, "G", "generation_power", [2000 if t != 12 else 2100 for t in range(168)]
    )
    check_output(st, "G", "num_units_on", [2 if t != 12 else 3 for t in range(168)])
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0 if t != 12 else 50 for t in range(168)])


def test_lp_version() -> None:
    """
    The optimal solution of the linear relaxation consists in producing exactly the
    demand at each hour. The number of on units is equal to the production divided by
    P_max.

    The optimal cost is then :
          50 x 2000 x 167 (prod step 1-12 and 14-168)
        + 50 x 2050 (prod step 13)
        + 2 x 1 x 167 (fixed cost step 1-12 and 14-168)
        + 2050/1000 x 1 (fixed cost step 13)
        + 0,05 x 50 (start up cost step 13)
        = 16 802 838,55
    """
    study = build_thermal_study(CASE_ID, "lp")
    config = optim_config_for(CASE_ID, "lp", 0, 167, [0])
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(16802838.55)
    check_output(
        st, "G", "generation_power", [2000 if t != 12 else 2050 for t in range(168)]
    )
    check_output(st, "G", "num_units_on", [2 if t != 12 else 2.05 for t in range(168)])
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_accurate_heuristic() -> None:
    """
    Solve the same problem as before with the accurate heuristic.

    The accurate heuristic is able to retrieve the milp optimal solution because when
    the number of on units found in the linear relaxation is ceiled, we found the
    optimal number of on units which is already feasible.
    """
    study = build_thermal_study(CASE_ID, "accurate")
    config = optim_config_for(CASE_ID, "accurate", 0, 167, [0])
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(16805387)
    check_output(
        st, "G", "generation_power", [2000 if t != 12 else 2100 for t in range(168)]
    )
    check_output(st, "G", "num_units_on", [2 if t != 12 else 3 for t in range(168)])
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0 if t != 12 else 50 for t in range(168)])


def test_fast_heuristic() -> None:
    """
    Solve the same problem as before with the fast heuristic.

    The optimal solution consists in having 3 units turned on between time steps 10
    and 19 with production equal to 2100 to respect pmin and 2 the rest of the time.
    Fast heuristic turns on 3 units for 10 timesteps because min down time is equal
    to 10.

    The optimal cost is then :
          50 x 2000 x 158 (prod step 1-9 and 20-168)
        + 50 x 2100 x 10 (prod step 10-19)
        = 16 850 000
    """
    study = build_thermal_study(CASE_ID, "fast")
    config = optim_config_for(CASE_ID, "fast", 0, 167, [0])
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(16850000)
    check_output(
        st,
        "G",
        "generation_power",
        [2000 if t not in range(10, 20) else 2100 for t in range(168)],
    )
    check_output(
        st,
        "G",
        "num_units_on",
        [2 if t not in range(10, 20) else 3 for t in range(168)],
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(
        st,
        "N",
        "spilled_energy",
        [0 if t not in range(10, 20) else (50 if t == 12 else 100) for t in range(168)],
    )
