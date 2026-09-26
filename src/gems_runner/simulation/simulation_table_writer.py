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
from typing import List, Literal, Optional, cast

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from gems_runner.simulation.simulation_table import SimulationColumns, SimulationTable

OutputFormat = Literal["csv", "parquet"]

# Parquet write settings, aligned with GEMS-ViewsBuilder (gems_views_builder/common.py)
PARQUET_COMPRESSION: Literal["zstd"] = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_SIZE = 64_000

# Suffix of the file holding rows shared by all scenarios (scenario_index empty),
# e.g. scenario-independent variables and the objective value of a frontal run.
COMMON_SCENARIO_SUFFIX = "scenario-common"

# Column types of the simulation table files. Fixed rather than inferred so
# that every file has the same schema, even when a column only holds empty
# values in some file (e.g. component in the common file), and the files can be
# read together (e.g. by GEMS-ViewsBuilder).
SIMULATION_TABLE_SCHEMA = pa.schema(
    [
        (SimulationColumns.BLOCK.value, pa.int64()),
        (SimulationColumns.COMPONENT.value, pa.string()),
        (SimulationColumns.OUTPUT.value, pa.string()),
        (SimulationColumns.ABSOLUTE_TIME_INDEX.value, pa.int64()),
        (SimulationColumns.BLOCK_TIME_INDEX.value, pa.int64()),
        (SimulationColumns.SCENARIO_INDEX.value, pa.int64()),
        (SimulationColumns.VALUE.value, pa.float64()),
        (SimulationColumns.BASIS_STATUS.value, pa.string()),
    ]
)


class SimulationTableWriter:
    """Writes a SimulationTable to disk as CSV or Parquet, one file per MC scenario.

    Each scenario goes to ``simulation_table_<id>_scenario-<N>.<ext>``; rows
    shared by all scenarios, if any, go to
    ``simulation_table_<id>_scenario-common.<ext>``.
    Every row is written exactly once, so the files can be concatenated back
    into the full table.
    """

    def __init__(self, output_format: OutputFormat = "csv") -> None:
        if output_format not in ("csv", "parquet"):
            raise ValueError(f"Unsupported output format: {output_format!r}")
        self.output_format = output_format

    def write(self, table: SimulationTable, output_dir: Path) -> List[Path]:
        """Write the full *table* into *output_dir*, split by scenario; return
        the written paths (none for an empty table)."""
        paths: List[Path] = []
        if table.data.empty:
            return paths
        for scenario, part in table.data.groupby(
            SimulationColumns.SCENARIO_INDEX.value, dropna=False, sort=True
        ):
            suffix = (
                COMMON_SCENARIO_SUFFIX
                if pd.isna(scenario)
                else f"scenario-{int(cast(float, scenario))}"
            )
            paths.append(self._write_part(part, output_dir, table.table_id, suffix))
        return paths

    def write_scenario(
        self, table: SimulationTable, output_dir: Path, scenario_id: Optional[int]
    ) -> Path:
        """Write the table of a single scenario to its ``scenario-<N>`` file, or,
        for *scenario_id* None, the rows shared by all scenarios to the
        ``scenario-common`` file. Safe to call concurrently for different
        scenarios."""
        suffix = (
            COMMON_SCENARIO_SUFFIX if scenario_id is None else f"scenario-{scenario_id}"
        )
        return self._write_part(table.data, output_dir, table.table_id, suffix)

    def _write_part(
        self, df: pd.DataFrame, output_dir: Path, table_id: str, suffix: str
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"simulation_table_{table_id}_{suffix}.{self.output_format}"
        # Both formats are written with pyarrow, which releases the GIL, so that
        # files can be written concurrently from several threads.
        # Pandas metadata is dropped so that the file only depends on the data
        # and the fixed schema, not on how the DataFrame was built.
        arrow_table = pa.Table.from_pandas(
            df, schema=SIMULATION_TABLE_SCHEMA, preserve_index=False
        ).replace_schema_metadata(None)
        if self.output_format == "parquet":
            pq.write_table(  # type: ignore[no-untyped-call]
                arrow_table,
                path,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=PARQUET_ROW_GROUP_SIZE,
            )
        else:
            pacsv.write_csv(arrow_table, path)
        return path
