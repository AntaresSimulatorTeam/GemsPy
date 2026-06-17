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

from math import ceil

import numpy as np
import pandas as pd


def get_max_unit_for_min_down_time(
    slot_length: int, max_units: pd.DataFrame, hours_in_week: int
) -> pd.DataFrame:
    max_units = shorten_df(max_units, hours_in_week)
    nb_units_max_min_down_time = shift_df(max_units, slot_length, hours_in_week)
    end_failures = max_units - shift_df(max_units, 1, hours_in_week)
    end_failures.where(end_failures > 0, 0, inplace=True)
    for j in range(slot_length):
        nb_units_max_min_down_time += shift_df(end_failures, j, hours_in_week)

    return nb_units_max_min_down_time


def shorten_df(df: pd.DataFrame, length_block: int) -> pd.DataFrame:
    blocks = len(df) // length_block
    df = df[: blocks * length_block]
    return df


def shift_df(df: pd.DataFrame, shift: int, length_block: int) -> pd.DataFrame:
    assert len(df) % length_block == 0
    blocks = len(df) // length_block
    values = df.values.reshape((blocks, length_block, df.shape[1]))
    shifted_values = np.roll(values, shift, axis=1)
    return pd.DataFrame(
        shifted_values.reshape((length_block * blocks, df.shape[1])),
        index=df.index,
    )


def get_max_failures(max_units: pd.DataFrame, hours_in_week: int) -> pd.DataFrame:
    max_units = shorten_df(max_units, hours_in_week)
    max_failures = shift_df(max_units, 1, hours_in_week) - max_units
    max_failures.where(max_failures > 0, 0, inplace=True)
    return max_failures


def get_max_unit(
    pmax: float, units: float, max_generating: pd.DataFrame
) -> pd.DataFrame:
    max_units = max_generating / pmax
    max_units = max_units.applymap(ceil)
    max_units.where(max_units < units, units, inplace=True)
    return max_units
