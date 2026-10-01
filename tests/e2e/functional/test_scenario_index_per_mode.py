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

"""
E2E test: scenario_index of scenario-independent outputs per resolution mode.

The 13_1 investment study is run with 2 MC scenarios whose load differs
(300 and 500). ``p_max`` (the invested capacity) is scenario-independent.

- frontal: both scenarios are solved in one problem, so ``p_max`` and the
  objective value are shared and keep an empty scenario_index.
- parallel-subproblems: each scenario is solved separately and sizes its own
  ``p_max``, so every row must carry the scenario it was solved for.
"""

import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from gems_runner.study.runner import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "13_1"

_CONFIG_HEADER = textwrap.dedent("""\
    time-scope:
      first-time-step: 0
      last-time-step: 0
    solver-options:
      name: highs
      logs: false
      parameters: ""
    scenario-scope:
      include:
        - 0
        - 1
""")

_PARALLEL_RESOLUTION = textwrap.dedent("""\
    resolution:
      mode: parallel-subproblems
      block-length: 1
""")


def _run_two_scenario_study(tmp_path: Path, resolution: str) -> pd.DataFrame:
    study_dir = tmp_path / "13_1"
    shutil.copytree(_STUDY_SRC, study_dir)

    system_path = study_dir / "input" / "system.yml"
    system_path.write_text(
        system_path.read_text().replace(
            """        - id: load
          time-dependent: false
          scenario-dependent: false
          value: 400""",
            """        - id: load
          time-dependent: true
          scenario-dependent: true
          value: load_ts""",
            1,
        )
    )
    data_series_dir = study_dir / "input" / "data-series"
    data_series_dir.mkdir()
    (data_series_dir / "load_ts.tsv").write_text("300\t500\n")
    (study_dir / "input" / "optim-config.yml").write_text(_CONFIG_HEADER + resolution)

    run_study(study_dir)

    output_files = sorted((study_dir / "output").glob("**/simulation_table_*.csv"))
    return pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)


def _rows(df: pd.DataFrame, output: str) -> pd.DataFrame:
    return df[df["output"] == output].sort_values("value")


def test_frontal_shared_rows_have_no_scenario(tmp_path: Path) -> None:
    df = _run_two_scenario_study(tmp_path, resolution="")

    p_max = _rows(df, "p_max")
    objective = _rows(df, "objective-value")
    assert len(p_max) == 1 and p_max["scenario_index"].isna().all()
    assert len(objective) == 1 and objective["scenario_index"].isna().all()


def test_parallel_rows_are_tagged_with_their_scenario(tmp_path: Path) -> None:
    df = _run_two_scenario_study(tmp_path, resolution=_PARALLEL_RESOLUTION)

    assert df["scenario_index"].notna().all()
    # Each scenario sizes its own capacity: the higher load needs more.
    p_max = _rows(df, "p_max")
    assert list(p_max["scenario_index"]) == [0, 1]
    assert list(p_max["value"]) == pytest.approx([100.0, 300.0])
    objective = _rows(df, "objective-value")
    assert list(objective["scenario_index"]) == [0, 1]
    assert list(objective["value"]) == pytest.approx([50_000.0, 132_000.0])
