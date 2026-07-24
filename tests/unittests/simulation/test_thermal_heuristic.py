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

import warnings

import pytest

from gems.simulation.thermal_heuristic import (
    find_min_generation_fast,
    find_nb_units_accurate,
)

# ---------------------------------------------------------------------------
# find_min_generation_fast — scalar cluster_max_generation broadcast
# ---------------------------------------------------------------------------

_GENERATION_POWER = [50.0, 100.0, 150.0, 100.0, 50.0, 0.0, 0.0, 50.0]


def test_find_min_generation_fast_scalar_matches_constant_list() -> None:
    nb_timesteps = len(_GENERATION_POWER)
    result_scalar = find_min_generation_fast(_GENERATION_POWER, 200.0, 50.0, 50.0, 2, 2)
    result_list = find_min_generation_fast(
        _GENERATION_POWER, [200.0] * nb_timesteps, 50.0, 50.0, 2, 2
    )
    assert result_scalar == result_list


def test_find_min_generation_fast_scalar_clamps_output() -> None:
    result = find_min_generation_fast(_GENERATION_POWER, 40.0, 50.0, 50.0, 2, 2)
    assert all(v <= 40.0 for v in result)


# ---------------------------------------------------------------------------
# find_nb_units_accurate — scalar nb_units_max broadcast
# ---------------------------------------------------------------------------

_NB_UNITS_ON_OPT = [1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0]


def test_find_nb_units_accurate_scalar_matches_constant_list() -> None:
    nb_timesteps = len(_NB_UNITS_ON_OPT)
    result_scalar = find_nb_units_accurate(_NB_UNITS_ON_OPT, 5, 2, 2)
    result_list = find_nb_units_accurate(_NB_UNITS_ON_OPT, [5.0] * nb_timesteps, 2, 2)
    assert result_scalar == result_list


# ---------------------------------------------------------------------------
# Non-integer min_up_duration / min_down_duration — warn and round
# ---------------------------------------------------------------------------


def test_find_min_generation_fast_warns_on_non_integer_min_up_duration() -> None:
    with pytest.warns(UserWarning, match="min_up_duration"):
        result_decimal = find_min_generation_fast(
            _GENERATION_POWER, 200.0, 50.0, 50.0, 2.6, 2
        )
    result_rounded = find_min_generation_fast(
        _GENERATION_POWER, 200.0, 50.0, 50.0, 3, 2
    )
    assert result_decimal == result_rounded


def test_find_min_generation_fast_no_warning_on_integer_duration() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        find_min_generation_fast(_GENERATION_POWER, 200.0, 50.0, 50.0, 2, 2)


def test_find_nb_units_accurate_warns_on_non_integer_min_down_duration() -> None:
    with pytest.warns(UserWarning, match="min_down_duration"):
        result_decimal = find_nb_units_accurate(_NB_UNITS_ON_OPT, 5, 2, 1.7)
    result_rounded = find_nb_units_accurate(_NB_UNITS_ON_OPT, 5, 2, 2)
    assert result_decimal == result_rounded
