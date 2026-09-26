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
E2E test: one simulation table file per scenario, on
the 13_1 investment study extended to 4 time steps and 2 MC scenarios.

- frontal: one file per scenario plus a scenario-common file holding the
  shared investment (p_max) and the objective value.
- sequential / parallel subproblems: each scenario is solved separately and
  written as soon as it is solved; every row belongs to a scenario, so no
  common file is written. The files must match splitting the full table.
"""

import shutil
import textwrap
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table import SimulationTable
from gems_runner.simulation.simulation_table_writer import SimulationTableWriter
from gems_runner.study.runner import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "13_1"

_CONFIG_HEADER = textwrap.dedent("""\
    time-scope:
      first-time-step: 0
      last-time-step: 3
    solver-options:
      name: highs
      logs: false
      parameters: ""
    scenario-scope:
      include:
        - 0
        - 1
""")

_RESOLUTIONS = {
    "frontal": "",
    "sequential": textwrap.dedent("""\
        resolution:
          mode: sequential-subproblems
          block-length: 2
        """),
    "parallel": textwrap.dedent("""\
        resolution:
          mode: parallel-subproblems
          block-length: 2
        """),
}


def _make_study(tmp_path: Path, mode: str) -> Path:
    """13_1 with a time- and scenario-dependent load (4 steps x 2 scenarios)."""
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
    (data_series_dir / "load_ts.tsv").write_text(
        "300\t500\n350\t450\n400\t550\n250\t480\n"
    )
    (study_dir / "input" / "optim-config.yml").write_text(
        _CONFIG_HEADER + _RESOLUTIONS[mode]
    )
    return study_dir


def _run_study_to_parquet(tmp_path: Path, mode: str) -> List[Path]:
    study_dir = _make_study(tmp_path, mode)
    run_study(study_dir, output_format="parquet")
    return sorted((study_dir / "output").glob("**/simulation_table_*.parquet"))


def _suffixes(paths: List[Path]) -> List[str]:
    return [p.stem.rsplit("_", 1)[-1] for p in paths]


def test_frontal_writes_common_file(tmp_path: Path) -> None:
    paths = _run_study_to_parquet(tmp_path, "frontal")

    assert _suffixes(paths) == ["scenario-0", "scenario-1", "scenario-common"]
    common = pd.read_parquet(paths[2])
    assert sorted(common["output"]) == ["objective-value", "p_max"]


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_separate_scenario_modes_write_no_common_file(
    tmp_path: Path, mode: str
) -> None:
    paths = _run_study_to_parquet(tmp_path, mode)

    assert _suffixes(paths) == ["scenario-0", "scenario-1"]
    for scenario, path in enumerate(paths):
        df = pd.read_parquet(path)
        assert (df["scenario_index"] == scenario).all()
        assert "p_max" in df["output"].values


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_streamed_files_match_splitting_the_full_table(
    tmp_path: Path, mode: str
) -> None:
    """Writing each scenario as soon as it is solved gives the same files as
    solving everything first and splitting the full table."""
    study_dir = _make_study(tmp_path, mode)
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    writer = SimulationTableWriter("parquet")

    full_table = SimulationSession(
        load_study(study_dir), optim_config, run_id="run"
    ).run()
    split_paths = writer.write(full_table, tmp_path / "split")

    streamed: Dict[int, SimulationTable] = {}
    session = SimulationSession(
        load_study(study_dir),
        optim_config,
        run_id="run",
        on_scenario_done=lambda scenario_id, table: streamed.update(
            {scenario_id: table}
        ),
    )
    assert session.run().data.empty  # nothing kept once handed to the callback
    streamed_paths = [
        writer.write_scenario(table, tmp_path / "streamed", scenario_id)
        for scenario_id, table in streamed.items()
    ]

    assert [p.name for p in split_paths] == [p.name for p in streamed_paths]
    for split_path, streamed_path in zip(split_paths, streamed_paths):
        pd.testing.assert_frame_equal(
            pd.read_parquet(split_path), pd.read_parquet(streamed_path)
        )


def test_split_files_can_be_read_together_like_views_builder(tmp_path: Path) -> None:
    """GEMS-ViewsBuilder concatenates all simulation table files with polars and
    finds non-time-dependent rows with is_null(): the files must share a schema
    and keep proper nulls."""
    pl = pytest.importorskip("polars")
    paths = _run_study_to_parquet(tmp_path, "frontal")

    df = pl.scan_parquet([str(p) for p in paths]).collect()
    objective = df.filter(pl.col("output") == "objective-value")
    assert objective.height == 1
    assert objective["absolute_time_index"].is_null().all()
    assert objective["scenario_index"].is_null().all()
