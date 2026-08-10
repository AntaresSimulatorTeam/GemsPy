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
End-to-end test for a thermal study mixing four different integer-strategies
in a single solve, using SimulationTable.

Study: tests/e2e/functional/studies/thermal_heuristic_four_clusters_mixed_strategies
Components:
  - N  : node (area model)
  - D  : fixed demand (load model, timeseries demand-ts, one day, 400-2600 MW)
  - G1 : thermal cluster, integer-strategy relaxed        (p_max=300, p_min=100, num_units=4, cost=10)
  - G2 : thermal_fast cluster, heuristic fast              (p_max=250, p_min=80,  num_units=3, cost=20)
  - G3 : thermal cluster, no integer-strategy (-> exact/MILP) (p_max=200, p_min=60, num_units=3, cost=30)
  - G4 : thermal cluster, heuristic accurate               (p_max=150, p_min=50,  num_units=2, cost=50)

Unlike every other case in this suite, the per-component integer-strategy is
fixed directly in system.yml rather than toggled uniformly across all thermal
components via a test-time "mode" — this is the only case that exercises a
relaxation, both heuristics, and plain MILP commitment side by side within one
solve. Bid costs increase from G1 (cheapest) to G4 (priciest, peaking unit),
so the demand cycle dispatches all four clusters at different load levels:
G1 alone overnight, G2/G3 joining through the day, G4 only at the afternoon
peak.

1 scenario x 1 day of 24 hours — enough to exercise every strategy at a
different load level without a multi-day study.
"""

import pytest

from gems_runner.session.session import SimulationSession
from tests.e2e.functional.thermal_heuristic_helpers import (
    build_thermal_study,
    check_output,
    optim_config_all_models,
    total_output_sum,
)

CASE_ID = "four_clusters_mixed_strategies"
THERMAL_COMPONENTS = ["G1", "G2", "G3", "G4"]

# fmt: off
_G1_GEN = [722.0, 547.0, 437.0, 400.0, 437.0, 547.0, 642.0, 870.0, 1135.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1055.0, 790.0]
_G2_GEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.0, 80.0, 80.0, 300.0, 585.0, 750.0, 750.0, 750.0, 750.0, 750.0, 750.0, 750.0, 750.0, 750.0, 585.0, 300.0, 160.0, 160.0]
_G3_GEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 328.0, 503.0, 563.0, 600.0, 563.0, 503.0, 328.0, 100.0, 0.0, 0.0, 0.0, 0.0]
_G4_GEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_G2_NODU = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0]
_G3_NODU = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
_G4_NODU = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# fmt: on


def test_mixed_integer_strategies() -> None:
    """
    Solve a single problem where G1 is relaxed, G2 uses the fast heuristic,
    G3 is plain MILP, and G4 uses the accurate heuristic — all at once.
    """
    study = build_thermal_study(CASE_ID, "milp")
    config = optim_config_all_models()
    st = SimulationSession(study, config).run()

    objective = st.data.loc[st.data["output"] == "objective-value", "value"].iloc[0]
    assert objective == pytest.approx(537680)

    check_output(st, "G1", "generation_power", _G1_GEN)
    check_output(st, "G2", "generation_power", _G2_GEN)
    check_output(st, "G3", "generation_power", _G3_GEN)
    check_output(st, "G4", "generation_power", _G4_GEN)

    # G1 is relaxed: continuous, and sits at exactly 4 units all day (its
    # cost makes it always worth running fully, and demand never dips below
    # what 4 units can produce at their minimum).
    check_output(st, "G1", "num_units_on", [4.0] * 24)
    check_output(st, "G2", "num_units_on", _G2_NODU)
    check_output(st, "G3", "num_units_on", _G3_NODU)
    check_output(st, "G4", "num_units_on", _G4_NODU)

    check_output(st, "N", "unsupplied_energy", [0.0] * 24)
    check_output(st, "N", "spilled_energy", [0.0] * 24)

    assert total_output_sum(st, THERMAL_COMPONENTS, "non_prop_cost") == pytest.approx(
        13665
    )
