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

import math
from typing import List

ZERO = 1e-6


def _compute_nb_units_from_window_max(
    window_region_start: int,
    window_region_end: int,
    window_size: int,
    nb_timesteps: int,
    nb_units_required: List[int],
    nb_units_on: List[int],
) -> float:
    """
    Fills nb_units_on in place and returns the total number of units assigned above nb_units_required.

    Border regions [0, window_region_start) and [window_region_end, nb_timesteps) are set to
    the global max of nb_units_required across both borders. The central region is processed
    in sliding windows of size window_size, each window taking its local max.

    Note: nb_units_on is modified even when this function is called during offset search —
    the caller is responsible for applying it once more with the best offset to get the
    final result.
    """
    total_excess_units = 0.0

    border_hours = list(range(window_region_start)) + list(
        range(window_region_end, nb_timesteps)
    )
    border_max = max((nb_units_required[h] for h in border_hours), default=0)
    for hour in border_hours:
        nb_units_on[hour] = border_max
        total_excess_units += border_max - nb_units_required[hour]

    for window_start in range(window_region_start, window_region_end, window_size):
        window_end = min(window_start + window_size, window_region_end)
        max_units = max(nb_units_required[window_start:window_end])
        for hour in range(window_start, window_end):
            nb_units_on[hour] = max_units
            total_excess_units += max_units - nb_units_required[hour]

    return total_excess_units


def find_nb_units_fast(
    generation_power: List[float],
    min_power_per_unit: float,
    max_power_per_unit: float,
    min_up_duration: int,
    min_down_duration: int,
) -> List[int]:
    """
    Fast heuristic: derives the number of running units from the optimised production
    timeseries. Timesteps are grouped into windows of size max(min_up_duration, min_down_duration)
    and each window is assigned the maximum unit count required within it. The window grid is
    shifted over all possible offsets and the offset that minimises the total number of
    units assigned above nb_units_required is retained.

    Parameters
    ----------
    generation_power:
        Optimal production per timestep (MW).
    min_power_per_unit:
        Minimum power output per unit (MW). If ~0, all units are considered off.
    max_power_per_unit:
        Maximum power output per unit (MW), used to convert production to unit count.
    min_up_duration:
        Minimum up-time (hours).
    min_down_duration:
        Minimum down-time (hours).

    Returns
    -------
    List[int]
        Number of running units per timestep.
    """
    nb_timesteps = len(generation_power)
    nb_units_on = [0] * nb_timesteps

    if abs(min_power_per_unit) < ZERO:
        return nb_units_on
    
    assert max_power_per_unit > ZERO

    nb_units_required = [math.ceil(p / max_power_per_unit) for p in generation_power]

    window_size = max(min_up_duration, min_down_duration)

    max_window_offset = max(min(window_size, nb_timesteps - window_size), 0)

    best_total_excess_units = float("inf")
    best_offset = 0
    for window_offset in range(max_window_offset + 1):
        total_excess_units = _compute_nb_units_from_window_max(
            window_region_start=window_offset,
            window_region_end=nb_timesteps - max_window_offset + window_offset,
            window_size=window_size,
            nb_timesteps=nb_timesteps,
            nb_units_required=nb_units_required,
            nb_units_on=nb_units_on,
        )
        if total_excess_units < best_total_excess_units:
            best_total_excess_units = total_excess_units
            best_offset = window_offset

    _compute_nb_units_from_window_max(
        window_region_start=best_offset,
        window_region_end=nb_timesteps - max_window_offset + best_offset,
        window_size=window_size,
        nb_timesteps=nb_timesteps,
        nb_units_required=nb_units_required,
        nb_units_on=nb_units_on,
    )

    return nb_units_on


def find_nb_units_accurate(
    production: List[float],
    nb_units_max: List[int],
    p_min: float,
    p_max: float,
    d_min_up: int,
    d_min_down: int,
    nb_units_max_min_down_time: List[int],
    max_failure: List[int],
) -> List[int]:
    """
    Accurate heuristic: derives the number of running units from the optimised production
    timeseries, enforcing min up/down time constraints.

    Parameters
    ----------
    production:
        Optimal production per timestep (MW).
    nb_units_max:
        Maximum number of available units per timestep.
    p_min:
        Minimum power output per unit (MW).
    p_max:
        Maximum power output per unit (MW).
    d_min_up:
        Minimum up-time (hours).
    d_min_down:
        Minimum down-time (hours).
    nb_units_max_min_down_time:
        Maximum units accounting for min-down-time constraint per timestep.
    max_failure:
        Maximum number of units that can fail (forced outages) per timestep.

    Returns
    -------
    List[int]
        Number of running units per timestep.
    """
    raise NotImplementedError
