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
End-to-end test for the thermal one-cluster study with P_min = 0 and a
single demand spike.

Same study as tests/e2e/functional/test_thermal_heuristic_one_cluster.py
(one node, one thermal cluster G, one fixed demand D), except that:
  - G's min_power_per_unit is overridden to 0 in memory (see
    ``thermal_heuristic_helpers.with_parameter_value``).
  - D's demand is overridden to 0 at every hour except the 13th (2050 MW),
    using the dedicated data series
    studies/thermal_heuristic_one_cluster/input/data-series/demand-ts-only-hour13.txt.

With demand at 0 everywhere else, the cluster only ever needs to be
committed around the 13th-hour spike, and P_min = 0 means it is never
forced to produce anything while idling. min_up_duration = 3 still forces
whichever units start for the spike to stay committed for the 13th, 14th
and 15th hour (all 3 start together, so startup_cost applies to all 3).
"""

import pytest

from gems_runner.session.session import SimulationSession
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_for,
)

CASE_ID = "one_cluster"
PARAMETER_OVERRIDES = {
    "G": {"min_power_per_unit": 0},
    "D": {"load": "demand-ts-only-hour13"},
}


def test_milp_version() -> None:
    """
    The 2050 MW spike at the 13th hour needs all 3 units. Since they all start together, startup_cost applies to
    all 3 (3 x 50). min_up_duration = 3 keeps them committed for the 13th,
    14th and 15th hour, but with P_min = 0 they produce nothing once
    demand drops back to 0.

    The optimal cost is then :
          50 x 2050 (prod step 13)
        + 3 x 1 (fixed cost step 13)
        + 3 x 50 (start up step 13, all 3 units)
        + 3 x 1 x 2 (fixed cost steps 14-15, still committed, P_min = 0 so no prod)
        = 102 659
    """
    study = build_thermal_study(CASE_ID, "milp", PARAMETER_OVERRIDES)
    config = optim_config_for("milp")
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(102659)
    check_output(
        st,
        "G",
        "non_prop_cost",
        [153 if t == 12 else (3 if t in (13, 14) else 0) for t in range(168)],
    )
    check_output(
        st, "G", "generation_power", [2050 if t == 12 else 0 for t in range(168)]
    )
    check_output(
        st, "G", "num_units_on", [3 if t in (12, 13, 14) else 0 for t in range(168)]
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_lp_version() -> None:
    """
    Same shape as the MILP solution but with a continuous unit count
    (2.05, matching 2050 / 1000 MW), still held committed for 3 hours by
    min_up_duration.
    """
    study = build_thermal_study(CASE_ID, "lp", PARAMETER_OVERRIDES)
    config = optim_config_for("lp")
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(102608.65)
    check_output(
        st,
        "G",
        "non_prop_cost",
        [104.55 if t == 12 else (2.05 if t in (13, 14) else 0) for t in range(168)],
    )
    check_output(
        st, "G", "generation_power", [2050 if t == 12 else 0 for t in range(168)]
    )
    check_output(
        st, "G", "num_units_on", [2.05 if t in (12, 13, 14) else 0 for t in range(168)]
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_accurate_heuristic() -> None:
    """
    Solve the same problem as before with the accurate heuristic.

    Ceiling the LP relaxation's 2.05 units at the 13th hour gives 3, which
    is already the MILP-optimal number of units, so the accurate heuristic
    retrieves the exact MILP solution.
    """
    study = build_thermal_study(CASE_ID, "accurate", PARAMETER_OVERRIDES)
    config = optim_config_for("accurate")
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(102659)
    check_output(
        st,
        "G",
        "non_prop_cost",
        [153 if t == 12 else (3 if t in (13, 14) else 0) for t in range(168)],
    )
    check_output(
        st, "G", "generation_power", [2050 if t == 12 else 0 for t in range(168)]
    )
    check_output(
        st, "G", "num_units_on", [3 if t in (12, 13, 14) else 0 for t in range(168)]
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)


def test_fast_heuristic() -> None:
    """
    Solve the same problem as before with the fast heuristic.

    The fast heuristic derives maximum_generation_power from a sliding window of
    size max(min_up_duration, min_down_duration) = max(3, 10) = 10 hours, so the 3
    committed units stay available for the whole 10-hour window containing the
    13th-hour spike (hours 11 to 20, 1-indexed), not just for min_up_duration. As
    with every other fast-heuristic test in this module, the reported objective
    only reflects the proportional generation cost, not the fixed/startup costs
    (``non_prop_cost``).

    The optimal cost is then :
          50 x 2050 (prod step 13)
        = 102 500
    """
    study = build_thermal_study(CASE_ID, "fast", PARAMETER_OVERRIDES)
    config = optim_config_for("fast")
    st = SimulationSession(study, config).run()

    assert st.data.loc[st.data["output"] == "objective-value", "value"].iloc[
        0
    ] == pytest.approx(102500)
    check_output(
        st,
        "G",
        "non_prop_cost",
        [153 if t == 10 else (3 if 11 <= t <= 19 else 0) for t in range(168)],
    )
    check_output(
        st, "G", "generation_power", [2050 if t == 12 else 0 for t in range(168)]
    )
    check_output(
        st, "G", "num_units_on", [3 if 10 <= t <= 19 else 0 for t in range(168)]
    )
    check_output(st, "N", "unsupplied_energy", [0.0] * 168)
    check_output(st, "N", "spilled_energy", [0.0] * 168)
