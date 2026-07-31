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
import warnings
from typing import List, Union

import linopy
import numpy as np
import xarray as xr

ZERO = 1e-6


def _round_to_int_duration(value: float, name: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > ZERO:
        warnings.warn(
            f"'{name}' should be a whole number of timesteps, got {value}; "
            f"rounding to {rounded}.",
            UserWarning,
            stacklevel=2,
        )
    return int(rounded)


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


def find_min_generation_fast(
    generation_power: List[float],
    cluster_max_generation: Union[float, List[float]],
    min_power_per_unit: float,
    max_power_per_unit: float,
    min_up_duration: float,
    min_down_duration: float,
) -> List[float]:
    """
    Fast heuristic: derives the minimum generation power from the optimised production
    timeseries. Timesteps are grouped into windows of size max(min_up_duration, min_down_duration)
    and each window is assigned the maximum unit count required within it. The window grid is
    shifted over all possible offsets and the offset that minimises the total number of
    units assigned above nb_units_required is retained. The result is clamped by
    cluster_max_generation.

    Parameters
    ----------
    generation_power:
        Optimal production per timestep (MW).
    cluster_max_generation:
        Maximum available generation (MW), either per timestep or as a single constant
        applied to every timestep.
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
    List[float]
        Minimum generation power per timestep (MW), clamped by cluster_max_generation.
    """
    min_up_duration = _round_to_int_duration(min_up_duration, "min_up_duration")
    min_down_duration = _round_to_int_duration(min_down_duration, "min_down_duration")
    nb_timesteps = len(generation_power)
    nb_units_on = [0] * nb_timesteps

    if isinstance(cluster_max_generation, (int, float)):
        cluster_max_generation = [float(cluster_max_generation)] * nb_timesteps

    if abs(min_power_per_unit) < ZERO:
        return [0.0] * nb_timesteps

    assert max_power_per_unit > ZERO

    nb_units_required = [math.ceil(p / max_power_per_unit) for p in generation_power]

    window_size = max(min_up_duration, min_down_duration, 1)

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

    return [
        min(n * min_power_per_unit, cluster_max_generation[t])
        for t, n in enumerate(nb_units_on)
    ]


def find_nb_units_accurate(
    nb_units_on_opt: List[float],
    nb_units_max: Union[float, List[float]],
    min_up_duration: float,
    min_down_duration: float,
) -> List[int]:
    """
    Accurate heuristic: enforces min up/down time constraints on the number of running
    units by solving a small LP.

    The initial lower bound is derived from the optimisation output by ceiling each value:
    nb_units_required[t] = ceil(nb_units_on_opt[t]).

    The LP minimises the total number of running units over the horizon subject to:
    - on-units dynamics: nb_on[t] - nb_on[t-1] = nb_starting[t] - nb_stopping[t]
    - forced outages bounded by simultaneous shutdowns
    - min up time: units started within the last min_up_duration hours must still be on
    - min down time: units stopped within the last min_down_duration hours cannot restart

    All indices are cyclic (weekly wrap-around).

    Parameters
    ----------
    nb_units_on_opt:
        Number of units on per timestep from the optimisation (float, before ceiling).
    nb_units_max:
        Maximum number of available units, either per timestep or as a single constant
        applied to every timestep.
    min_up_duration:
        Minimum consecutive hours a unit must stay on after starting.
    min_down_duration:
        Minimum consecutive hours a unit must stay off after stopping.

    Returns
    -------
    List[int]
        Number of running units per timestep respecting min up/down time constraints.

    Raises
    ------
    AssertionError
        If the LP has no feasible solution.
    """
    min_up_duration = _round_to_int_duration(min_up_duration, "min_up_duration")
    min_down_duration = _round_to_int_duration(min_down_duration, "min_down_duration")
    if isinstance(nb_units_max, (int, float)):
        nb_units_max = [nb_units_max] * len(nb_units_on_opt)
    # round before ceil to absorb LP solver numerical noise (e.g. 10.0000001 → 10, not 11)
    nb_units_required = [float(math.ceil(round(v, 6))) for v in nb_units_on_opt]

    # Variables per timestep: nb_on, nb_outages, nb_starting, nb_stopping
    problem = linopy.Model()
    timesteps = range(len(nb_units_on_opt))

    nb_on_var = problem.add_variables(
        lower=np.array(nb_units_required),
        upper=np.array(nb_units_max),
        coords=[timesteps],
        name="nb_on",
    )
    nb_outages_var = problem.add_variables(
        lower=0.0,
        upper=np.array(
            [max(nb_units_max[t - 1] - nb_units_max[t], 0) for t in timesteps]
        ),
        coords=[timesteps],
        name="nb_outages",
    )
    nb_starting_var = problem.add_variables(
        lower=0.0, coords=[timesteps], name="nb_starting"
    )
    nb_stopping_var = problem.add_variables(
        lower=0.0, coords=[timesteps], name="nb_stopping"
    )

    problem.objective = nb_on_var.sum()

    # On-units dynamics: nb_on[t] - nb_on[t-1] - nb_starting[t] + nb_stopping[t] = 0
    problem.add_constraints(
        nb_on_var - nb_on_var.roll(dim_0=1) - nb_starting_var + nb_stopping_var == 0,
        name="on_units_dynamics",
    )

    # Forced outages bounded by simultaneous shutdowns: nb_outages[t] <= nb_stopping[t]
    problem.add_constraints(nb_outages_var - nb_stopping_var <= 0, name="outages_bound")

    # Min up time: nb_on[t] >= sum(nb_starting[k] - nb_outages[k]) over last min_up_duration hours
    problem.add_constraints(
        nb_on_var
        - sum(
            (nb_starting_var - nb_outages_var).roll(dim_0=shift)
            for shift in range(min_up_duration)
        )
        >= 0,
        name="min_up",
    )

    # Min down time: nb_on[t] + sum(nb_stopping[k] for k in last min_down_duration hours)
    #   <= nb_units_max[t - min_down_duration] + sum(positive increments of nb_units_max over that window)
    problem.add_constraints(
        nb_on_var
        + sum(nb_stopping_var.roll(dim_0=shift) for shift in range(min_down_duration))
        <= xr.DataArray(np.array(nb_units_max, dtype=float), dims=["dim_0"]).roll(
            dim_0=min_down_duration
        )
        + sum(
            xr.DataArray(
                np.maximum(np.diff(nb_units_max, prepend=nb_units_max[-1]), 0).astype(
                    float
                ),
                dims=["dim_0"],
            ).roll(dim_0=shift)
            for shift in range(min_down_duration)
        ),
        name="min_down",
    )

    problem.solve(solver_name="highs")

    assert (
        problem.status == "ok"
    ), "Accurate thermal heuristic LP has no feasible solution."
    solution = nb_on_var.solution.values
    return [math.ceil(float(solution[t])) for t in timesteps]
