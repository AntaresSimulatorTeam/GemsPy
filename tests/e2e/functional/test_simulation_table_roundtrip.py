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

"""E2E test: a simulation table saved to disk and reopened returns the same
values as the table in memory, through every accessor, regardless of row order.

Runs cover frontal, sequential with overlap and parallel modes, each with one
and with several scenarios, saved in CSV and in Parquet.
"""

from pathlib import Path
from typing import Callable, Dict

import pytest

from gems_craft.optim_config.parsing import ResolutionMode
from gems_craft.study.parsing import OutputFormat
from gems_runner.simulation.simulation_table import SimulationTable
from tests.e2e.functional.simulation_table_output import assert_same_tables
from tests.e2e.functional.test_simulation_table_views import (
    _run_13_1,
    _sequential_overlap,
)

_RUNS: Dict[str, Callable[[], SimulationTable]] = {
    "frontal-1": lambda: _run_13_1(ResolutionMode.FRONTAL, [0]),
    "frontal-2": lambda: _run_13_1(ResolutionMode.FRONTAL, [0, 1]),
    "sequential-overlap-1": _sequential_overlap,
    "sequential-overlap-2": lambda: _run_13_1(
        ResolutionMode.SEQUENTIAL_SUBPROBLEMS, [0, 1], block_length=2, block_overlap=1
    ),
    "parallel-1": lambda: _run_13_1(
        ResolutionMode.PARALLEL_SUBPROBLEMS, [0], block_length=2
    ),
    "parallel-2": lambda: _run_13_1(
        ResolutionMode.PARALLEL_SUBPROBLEMS, [0, 1], block_length=2
    ),
}


@pytest.mark.parametrize("fmt", [OutputFormat.PARQUET, OutputFormat.CSV])
@pytest.mark.parametrize("run", list(_RUNS))
def test_saved_table_reopens_with_the_same_values(
    tmp_path: Path, run: str, fmt: OutputFormat
) -> None:
    memory = _RUNS[run]()

    stored = SimulationTable.open(memory.save(tmp_path / "table", fmt))

    assert_same_tables(memory, stored)
