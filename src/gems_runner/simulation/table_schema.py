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

"""Columns and schema of the simulation table, shared by the table, its views
and its storages."""

from enum import Enum
from typing import Any, Dict, Literal

import pandas as pd
import pyarrow as pa


class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute_time_index"
    BLOCK_TIME_INDEX = "block_time_index"
    SCENARIO_INDEX = "scenario_index"
    VALUE = "value"
    BASIS_STATUS = "basis_status"


_BLOCK = SimulationColumns.BLOCK.value
_COMPONENT = SimulationColumns.COMPONENT.value
_OUTPUT = SimulationColumns.OUTPUT.value
_TIME = SimulationColumns.ABSOLUTE_TIME_INDEX.value
_SCENARIO = SimulationColumns.SCENARIO_INDEX.value
_VALUE = SimulationColumns.VALUE.value

OBJECTIVE_VALUE_OUTPUT = "objective-value"

PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 64_000

# One schema for the simulation table, whatever the content: nullable integer
# indices (empty for a missing dimension), string labels and float values.
_INDEX_COLUMNS = {
    SimulationColumns.BLOCK,
    SimulationColumns.ABSOLUTE_TIME_INDEX,
    SimulationColumns.BLOCK_TIME_INDEX,
    SimulationColumns.SCENARIO_INDEX,
}
_LABEL_COLUMNS = {
    SimulationColumns.COMPONENT,
    SimulationColumns.OUTPUT,
    SimulationColumns.BASIS_STATUS,
}
SIMULATION_TABLE_SCHEMA = pa.schema(
    [
        (
            column.value,
            (
                pa.int64()
                if column in _INDEX_COLUMNS
                else pa.string() if column in _LABEL_COLUMNS else pa.float64()
            ),
        )
        for column in SimulationColumns
    ]
)
# Labels use the python-backed string dtype: with the pyarrow-backed one,
# combining a label comparison and an index comparison with ``&`` raises on
# empty cells (e.g. the component of the objective-value row).
SIMULATION_TABLE_DTYPES: Dict[str, Any] = {
    column.value: (
        "Int64"
        if column in _INDEX_COLUMNS
        else pd.StringDtype("python") if column in _LABEL_COLUMNS else "float64"
    )
    for column in SimulationColumns
}


def _apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Give *df* the columns and dtypes of the simulation table schema."""
    return df[list(SIMULATION_TABLE_DTYPES)].astype(SIMULATION_TABLE_DTYPES)


def _with_schema(df: pd.DataFrame) -> pd.DataFrame:
    """*df* itself if it already has the schema's columns, in order, and dtypes;
    otherwise a converted copy. Frames built by the session already do."""
    if df.dtypes.to_dict() == SIMULATION_TABLE_DTYPES and list(df.columns) == list(
        SIMULATION_TABLE_DTYPES
    ):
        return df
    return _apply_schema(df)


def _empty_frame() -> pd.DataFrame:
    """A table without rows, with the columns and dtypes of the schema."""
    return pd.DataFrame(
        {
            name: pd.Series(dtype=dtype)
            for name, dtype in SIMULATION_TABLE_DTYPES.items()
        }
    )


def _equals(column: "pd.Series[Any]", value: Any) -> "pd.Series[bool]":
    """``column == value`` as a plain boolean mask: empty cells never match."""
    return column.eq(value).fillna(False).astype(bool)


def _same_key(column: "pd.Series[Any]", value: Any) -> "pd.Series[bool]":
    """Like ``_equals``, but an empty *value* matches the empty cells."""
    return column.isna() if pd.isna(value) else _equals(column, value)
