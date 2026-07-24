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

from pathlib import Path

import numpy as np
import pytest

from gems_runner.simulation.thermal_heuristic import find_min_generation_fast

DATA_DIR = Path(__file__).parent / "data/thermal_heuristic_fast_min_down_not_respected"


def test_fast_heuristic() -> None:
    """
    Check that find_min_generation_fast produces the correct min_generating for a cluster
    with very long d_min_up (72h) and d_min_down (30h). The fast heuristic uses
    window-based scheduling and doesn't strictly enforce these constraints.
    """
    generation_power = np.loadtxt(DATA_DIR / "itr1_fast_cluster.txt").tolist()

    min_generation = find_min_generation_fast(
        generation_power=generation_power,
        cluster_max_generation=22064,
        min_power_per_unit=870.21,
        max_power_per_unit=1318.5,
        min_up_duration=72,
        min_down_duration=30,
    )

    expected_output = np.loadtxt(DATA_DIR / "itr2_fast_cluster.txt")
    assert min_generation == pytest.approx(expected_output)
