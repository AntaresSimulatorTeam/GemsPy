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

from gems.simulation.thermal_heuristic import (
    find_min_generation_fast,
    find_nb_units_accurate,
)
from gems.study.data import ComponentParameterIndex
from gems.study.folder import load_study

DATA_DIR = Path(__file__).parent / "data/thermal_heuristic_six_clusters"
STUDY_DIR = Path(__file__).parent / "studies/thermal_heuristic_six_clusters"
CLUSTERS = ["G1", "G2", "G3", "G4", "G5", "G6"]


@pytest.fixture
def data_path() -> Path:
    return DATA_DIR


def test_accurate_heuristic() -> None:
    """
    Check that find_nb_units_accurate produces the same nb_units_on as Antares.

    Input : itr1_accurate_cluster*.txt  — LP fractional nb_on values per timestep.
    Output: itr2_accurate_cluster*.txt  — integer nb_on after the heuristic,
                                          respecting min_up/min_down constraints.
    """
    number_hours = 168
    scenario = 0

    study = load_study(STUDY_DIR)

    for j, cluster in enumerate(CLUSTERS):
        max_power_per_unit = float(
            study.database.get_value(
                ComponentParameterIndex(cluster, "max_power_per_unit"), 0, scenario
            )
        )
        min_up_duration = int(
            study.database.get_value(
                ComponentParameterIndex(cluster, "min_up_duration"), 0, scenario
            )
        )
        min_down_duration = int(
            study.database.get_value(
                ComponentParameterIndex(cluster, "min_down_duration"), 0, scenario
            )
        )

        # itr1: LP fractional nb_on values
        nb_units_on_opt = np.loadtxt(
            DATA_DIR / f"accurate/itr1_accurate_cluster{j+1}.txt"
        ).tolist()

        # nb_units_max[t] = ceil(cluster_max_generation[t] / max_power_per_unit)
        nb_units_max = [
            int(
                np.ceil(
                    study.database.get_value(
                        ComponentParameterIndex(cluster, "cluster_max_generation"),
                        t,
                        scenario,
                    )
                    / max_power_per_unit
                )
            )
            for t in range(number_hours)
        ]

        nb_units_on = find_nb_units_accurate(
            nb_units_on_opt=nb_units_on_opt,
            nb_units_max=nb_units_max,
            min_up_duration=min_up_duration,
            min_down_duration=min_down_duration,
        )

        # itr2: expected integer nb_on after heuristic
        expected_output = np.loadtxt(
            DATA_DIR / f"accurate/itr2_accurate_cluster{j+1}.txt"
        )
        for t in range(number_hours):
            assert nb_units_on[t] == pytest.approx(expected_output[t])


def test_fast_heuristic() -> None:
    """
    Check that find_min_generation_fast produces the same min_generating as Antares with
    the same LP production input.
    """
    number_hours = 168
    scenario = 0

    study = load_study(STUDY_DIR)

    for j, cluster in enumerate(CLUSTERS):
        max_power_per_unit = float(
            study.database.get_value(
                ComponentParameterIndex(cluster, "max_power_per_unit"), 0, scenario
            )
        )
        p_min = float(
            study.database.get_value(
                ComponentParameterIndex(cluster, "min_power_per_unit"), 0, scenario
            )
        )
        min_up_duration = int(
            study.database.get_value(
                ComponentParameterIndex(cluster, "min_up_duration"), 0, scenario
            )
        )
        min_down_duration = int(
            study.database.get_value(
                ComponentParameterIndex(cluster, "min_down_duration"), 0, scenario
            )
        )
        cluster_max_generation = [
            float(
                study.database.get_value(
                    ComponentParameterIndex(cluster, "cluster_max_generation"),
                    t,
                    scenario,
                )
            )
            for t in range(number_hours)
        ]

        generation_power = np.loadtxt(
            DATA_DIR / f"fast/itr1_fast_cluster{j+1}.txt"
        ).tolist()

        min_generation = find_min_generation_fast(
            generation_power=generation_power,
            cluster_max_generation=cluster_max_generation,
            min_power_per_unit=p_min,
            max_power_per_unit=max_power_per_unit,
            min_up_duration=min_up_duration,
            min_down_duration=min_down_duration,
        )

        expected_output = np.loadtxt(DATA_DIR / f"fast/itr2_fast_cluster{j+1}.txt")
        assert min_generation == pytest.approx(expected_output)
