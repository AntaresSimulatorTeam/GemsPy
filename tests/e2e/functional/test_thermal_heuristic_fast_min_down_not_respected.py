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

from gems.simulation.thermal_heuristic import find_min_generation_fast
from gems.study.data import ComponentParameterIndex
from gems.study.folder import load_study

DATA_DIR = Path(__file__).parent / "data/thermal_heuristic_fast_min_down_not_respected"
STUDY_DIR = (
    Path(__file__).parent / "studies/thermal_heuristic_fast_min_down_not_respected"
)
CLUSTER = "G1"


def test_fast_heuristic() -> None:
    """
    Check that find_min_generation_fast produces the correct min_generating for a cluster
    with very long d_min_up (72h) and d_min_down (30h). The fast heuristic uses
    window-based scheduling and doesn't strictly enforce these constraints.
    """
    number_hours = 168
    scenario = 0

    study = load_study(STUDY_DIR)

    p_max = float(
        study.database.get_value(
            ComponentParameterIndex(CLUSTER, "max_power_per_unit"), 0, scenario
        )
    )
    p_min = float(
        study.database.get_value(
            ComponentParameterIndex(CLUSTER, "min_power_per_unit"), 0, scenario
        )
    )
    d_min_up = int(
        study.database.get_value(
            ComponentParameterIndex(CLUSTER, "min_up_duration"), 0, scenario
        )
    )
    d_min_down = int(
        study.database.get_value(
            ComponentParameterIndex(CLUSTER, "min_down_duration"), 0, scenario
        )
    )
    cluster_max_generation = [
        float(
            study.database.get_value(
                ComponentParameterIndex(CLUSTER, "cluster_max_generation"), t, scenario
            )
        )
        for t in range(number_hours)
    ]

    generation_power = np.loadtxt(DATA_DIR / "itr1_fast_cluster.txt").tolist()

    min_generation = find_min_generation_fast(
        generation_power=generation_power,
        cluster_max_generation=cluster_max_generation,
        min_power_per_unit=p_min,
        max_power_per_unit=p_max,
        min_up_duration=d_min_up,
        min_down_duration=d_min_down,
    )

    expected_output = np.loadtxt(DATA_DIR / "itr2_fast_cluster.txt")
    assert min_generation == pytest.approx(expected_output)
