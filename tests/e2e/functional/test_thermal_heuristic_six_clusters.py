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

from pathlib import Path

import numpy as np
import pytest

from gems_runner.simulation.thermal_heuristic import (
    find_min_generation_fast,
    find_num_units_accurate,
)

DATA_DIR = Path(__file__).parent / "data/thermal_heuristic_six_clusters"
NUMBER_HOURS = 168
SCENARIO = 0

# cluster -> (max_power_per_unit, min_power_per_unit, min_up_duration, min_down_duration)
CLUSTERS = {
    "G1": (64, 32, 3, 3),
    "G2": (221, 111, 3, 3),
    "G3": (486, 194, 2, 2),
    "G4": (218, 87, 1, 1),
    "G5": (29, 14, 1, 1),
    "G6": (159, 80, 3, 3),
}


def _cluster_max_generation(cluster: str) -> list:
    series = np.loadtxt(DATA_DIR / f"series_{cluster}.txt")
    return series[:NUMBER_HOURS, SCENARIO].tolist()


def test_accurate_heuristic() -> None:
    """
    Check that find_num_units_accurate produces the expected reference result.

    Input : itr1_accurate_cluster*.txt  — LP fractional num_on values per timestep.
    Output: itr2_accurate_cluster*.txt  — integer num_on after the heuristic,
                                          respecting min_up/min_down constraints.
    """
    for j, (
        cluster,
        (max_power_per_unit, _, min_up_duration, min_down_duration),
    ) in enumerate(CLUSTERS.items()):
        # itr1: LP fractional num_on values
        num_units_on_opt = np.loadtxt(
            DATA_DIR / f"accurate/itr1_accurate_cluster{j+1}.txt"
        ).tolist()

        # num_units_max[t] = ceil(cluster_max_generation[t] / max_power_per_unit)
        num_units_max = [
            np.ceil(v / max_power_per_unit) for v in _cluster_max_generation(cluster)
        ]

        num_units_on = find_num_units_accurate(
            num_units_on_opt=num_units_on_opt,
            num_units_max=num_units_max,
            min_up_duration=min_up_duration,
            min_down_duration=min_down_duration,
            solver_name="highs",
        )

        # itr2: expected integer num_on after heuristic
        expected_output = np.loadtxt(
            DATA_DIR / f"accurate/itr2_accurate_cluster{j+1}.txt"
        )
        for t in range(NUMBER_HOURS):
            assert num_units_on[t] == pytest.approx(expected_output[t])


def test_fast_heuristic() -> None:
    """
    Check that find_min_generation_fast produces the expected reference min_generating
    values for the same LP production input.
    """
    for j, (
        cluster,
        (max_power_per_unit, p_min, min_up_duration, min_down_duration),
    ) in enumerate(CLUSTERS.items()):
        generation_power = np.loadtxt(
            DATA_DIR / f"fast/itr1_fast_cluster{j+1}.txt"
        ).tolist()

        min_generation = find_min_generation_fast(
            generation_power=generation_power,
            cluster_max_generation=_cluster_max_generation(cluster),
            min_power_per_unit=p_min,
            max_power_per_unit=max_power_per_unit,
            min_up_duration=min_up_duration,
            min_down_duration=min_down_duration,
        )

        expected_output = np.loadtxt(DATA_DIR / f"fast/itr2_fast_cluster{j+1}.txt")
        assert min_generation == pytest.approx(expected_output)
