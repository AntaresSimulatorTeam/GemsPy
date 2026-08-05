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
End-to-end tests for the thermal one-cluster-with-ramp study.

Study: tests/e2e/functional/studies/thermal_heuristic_one_cluster_with_ramp
Components:
  - N : node (area model, spillage_cost=100, unsupplied_energy_cost=10000)
  - D : fixed demand (ramps 0->300 MW over 7 hours, then back to 0 at t=12, zeros thereafter)
  - G : thermal cluster (p_max=500, p_min=100, num_units_max=3, market_bid_cost=10, startup_cost=10,
                         d_min_up=1, d_min_down=1) with ramping constraints (ramp_up=30, ramp_down=30)

The study is solved over a full week (168 timesteps) but only the first 12-13 timesteps
are asserted, since that is the window where the demand ramp (and therefore the
interesting heuristic behavior) takes place; all remaining timesteps are flat at zero.

Three variants of the same study are compared:
  - test_milp_version: reference solution with the full MILP (integer num_units_on/
    starting/stopping), used as the ground truth for objective value and dispatch.
  - test_accurate_heuristic: the accurate heuristic's second LP has no integer
    constraint on the number of starting/stopping units, so num_units_on ends up
    fractional even though the relaxed LP is otherwise feasible.
  - test_fast_heuristic: the fast heuristic rounds num_units_on to integers (NODU=1
    over the ramp window), but a single unit ramping 100 MW in one step exceeds the
    30 MW/unit ramp constraint, so the resulting dispatch is ramp-infeasible even
    though the solver reports an optimal LP solution.
"""

import pytest

from gems_runner.session.session import SimulationSession
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_for,
)

CASE_ID = "one_cluster_with_ramp"


def test_milp_version() -> None:
    """Solve weekly problem with one cluster and ramp constraints with milp."""
    study = build_thermal_study(CASE_ID, "milp")
    config = optim_config_for(CASE_ID, "milp", 0, 12, [0])
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(29040)

    check_output(
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
            0.0,
        ],
    )
    check_output(
        st,
        "G",
        "num_units_on",
        [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0],
    )
    check_output(st, "N", "spilled_energy", [0.0, 50.0] + [0.0] * 9 + [50.0, 0.0])
    check_output(st, "N", "unsupplied_energy", [0.0] * 13)


def test_accurate_heuristic() -> None:
    """
    With the ramp model, the accurate heuristic's second LP produces fractional
    num_units_on values, because it has no integer constraint on the number of
    starting/stopping units.
    """
    study = build_thermal_study(CASE_ID, "accurate")
    config = optim_config_for(CASE_ID, "accurate", 0, 12, [0])
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(29011.4616736)

    check_output(
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
        abs=1e-2,
    )  # non-integer values, as expected from the accurate heuristic's relaxed LP
    check_output(
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
        abs=1e-2,
    )
    check_output(
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
        abs=1e-2,
    )


def test_fast_heuristic() -> None:
    """
    With the ramp model, the fast heuristic rounds num_units_on to 1 over the demand
    ramp, but a single unit ramping 100 MW in one step violates the ramp constraint of
    30 MW/unit: the resulting dispatch is ramp-infeasible even though the LP reports
    an optimal solution.
    """
    study = build_thermal_study(CASE_ID, "fast")
    config = optim_config_for(CASE_ID, "fast", 0, 12, [0])
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(29000)

    check_output(
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
            0.0,
        ],
    )
    check_output(st, "G", "num_units_on", [0.0] + [1.0] * 11 + [0.0])
