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

"""Reading the simulation table that run_study writes to disk."""

from pathlib import Path

import pandas as pd

from gems_runner.session.session import SIMULATION_TABLE_FOLDER
from gems_runner.simulation.simulation_table import SimulationTable


def read_run_table(study_dir: Path) -> pd.DataFrame:
    """All rows of the table folder of the single run of *study_dir*."""
    run_dirs = sorted((study_dir / "output").iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run folder, got {len(run_dirs)}"
    table = SimulationTable.open(run_dirs[0] / SIMULATION_TABLE_FOLDER)
    return table.to_pandas(load_all=True)
