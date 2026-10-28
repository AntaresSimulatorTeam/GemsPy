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

"""Reading the simulation table that run_study writes to disk, and comparing
a stored table with the same table in memory."""

from pathlib import Path

import pandas as pd

from gems_runner.session.session import SIMULATION_TABLE_FOLDER
from gems_runner.simulation.simulation_table import SimulationTable

_KEY = ["block", "component", "output", "absolute_time_index", "scenario_index"]


def read_run_table(study_dir: Path) -> pd.DataFrame:
    """All rows of the table folder of the single run of *study_dir*."""
    run_dirs = sorted((study_dir / "output").iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run folder, got {len(run_dirs)}"
    table = SimulationTable.open(run_dirs[0] / SIMULATION_TABLE_FOLDER)
    return table.to_pandas(load_all=True)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(_KEY, na_position="first").reset_index(drop=True)


def assert_same_tables(memory: SimulationTable, stored: SimulationTable) -> None:
    """Compare every accessor of two tables, regardless of row order."""
    data = memory.data
    pd.testing.assert_frame_equal(
        _sorted(stored.to_pandas(load_all=True)), _sorted(data)
    )
    pd.testing.assert_frame_equal(
        stored.objective_values()
        .sort_values(["block", "scenario_index"])
        .reset_index(drop=True),
        memory.objective_values()
        .sort_values(["block", "scenario_index"])
        .reset_index(drop=True),
    )
    assert stored.scenarios == memory.scenarios
    assert stored.table_id == memory.table_id

    stored_parts = list(stored.iter_scenario_frames())
    memory_parts = list(memory.iter_scenario_frames())
    assert [s for s, _ in stored_parts] == [s for s, _ in memory_parts]
    for (_, a), (_, b) in zip(stored_parts, memory_parts):
        pd.testing.assert_frame_equal(_sorted(a), _sorted(b))

    outputs = data[data["component"].notna()][["component", "output", "block"]]
    for component, output, block in outputs.drop_duplicates().itertuples(index=False):
        # A fresh view per call, so that on disk the scenario reads are used.
        pd.testing.assert_frame_equal(
            stored.component(component).output(output, block=block).data,
            memory.component(component).output(output, block=block).data,
        )
        for scenario in memory.scenarios:
            pd.testing.assert_series_equal(
                stored.component(component)
                .output(output, block=block)
                .value(scenario_index=scenario),
                memory.component(component)
                .output(output, block=block)
                .value(scenario_index=scenario),
            )
