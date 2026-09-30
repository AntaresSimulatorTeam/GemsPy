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
  common file is written.
- In every mode the files must be byte-identical to splitting the full table;
  in frontal mode each scenario's rows are built on demand after the solve.
"""

import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
)
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
          block-overlap: 1
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


def _stream(
    study_dir: Path,
) -> Tuple[SimulationTable, Dict[Optional[int], SimulationTable]]:
    """Run with on_scenario_done and collect the handed-over tables."""
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    streamed: Dict[Optional[int], SimulationTable] = {}

    def collect(scenario_id: Optional[int], table: SimulationTable) -> None:
        assert scenario_id not in streamed, "scenario handed over twice"
        streamed[scenario_id] = table

    returned = SimulationSession(
        load_study(study_dir),
        optim_config,
        run_id="run",
        on_scenario_done=collect,
    ).run()
    return returned, streamed


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_streamed_files_match_splitting_the_full_table(
    tmp_path: Path, mode: str, output_format: str
) -> None:
    """Handing results over one scenario at a time gives byte-identical files
    to solving everything first and splitting the full table."""
    study_dir = _make_study(tmp_path, mode)
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    writer = SimulationTableWriter(output_format)  # type: ignore[arg-type]

    full_table = SimulationSession(
        load_study(study_dir), optim_config, run_id="run"
    ).run()
    split_paths = sorted(writer.write(full_table, tmp_path / "split"))

    returned, streamed = _stream(study_dir)
    assert returned.data.empty  # nothing kept once handed to the callback
    streamed_paths = sorted(
        writer.write_scenario(table, tmp_path / "streamed", scenario_id)
        for scenario_id, table in streamed.items()
    )

    assert [p.name for p in split_paths] == [p.name for p in streamed_paths]
    for split_path, streamed_path in zip(split_paths, streamed_paths, strict=True):
        assert split_path.read_bytes() == streamed_path.read_bytes(), split_path.name


def test_frontal_hands_over_common_rows_once_and_each_scenario_once(
    tmp_path: Path,
) -> None:
    _, streamed = _stream(_make_study(tmp_path, "frontal"))

    assert sorted(streamed, key=lambda s: -1 if s is None else s) == [None, 0, 1]
    assert sorted(streamed[None].data["output"]) == ["objective-value", "p_max"]
    for scenario_id in (0, 1):
        scenario_col = streamed[scenario_id].data["scenario_index"]
        assert (scenario_col == scenario_id).all()


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


def test_sequential_overlap_keeps_one_row_per_block(tmp_path: Path) -> None:
    """With block-overlap, a time step solved in two blocks appears once per
    block in its scenario file."""
    paths = _run_study_to_parquet(tmp_path, "sequential")

    df = pd.read_parquet(paths[0])
    generation = df[
        (df["output"] == "generation")
        & (df["component"] == "already_installed_generator")
    ]
    blocks_per_time_step = generation.groupby("absolute_time_index")["block"].apply(
        sorted
    )
    assert blocks_per_time_step.to_dict() == {0: [0], 1: [0, 1], 2: [1, 2], 3: [2]}


def test_frontal_non_consecutive_scenario_ids(tmp_path: Path) -> None:
    """Scenario ids are mapped to their position in the solution arrays: with
    include [0, 2], the second position is scenario 2."""
    study_dir = _make_study(tmp_path, "frontal")
    (study_dir / "input" / "data-series" / "load_ts.tsv").write_text(
        "300\t500\t700\n350\t450\t650\n400\t550\t720\n250\t480\t690\n"
    )
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(_CONFIG_HEADER.replace("    - 1\n", "    - 2\n"))
    optim_config = load_optim_config(config_path)
    assert optim_config is not None
    assert optim_config.scenario_scope.scenario_ids == [0, 2]
    writer = SimulationTableWriter("parquet")

    full_table = SimulationSession(
        load_study(study_dir), optim_config, run_id="run"
    ).run()
    split_paths = sorted(writer.write(full_table, tmp_path / "split"))
    _, streamed = _stream(study_dir)
    streamed_paths = sorted(
        writer.write_scenario(table, tmp_path / "streamed", scenario_id)
        for scenario_id, table in streamed.items()
    )

    assert _suffixes(streamed_paths) == ["scenario-0", "scenario-2", "scenario-common"]
    assert [p.name for p in split_paths] == [p.name for p in streamed_paths]
    for split_path, streamed_path in zip(split_paths, streamed_paths, strict=True):
        assert split_path.read_bytes() == streamed_path.read_bytes(), split_path.name
    scenario_2 = pd.read_parquet(streamed_paths[1])
    assert (scenario_2["scenario_index"] == 2).all()


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_each_scenario_is_handed_over_before_the_next_is_solved(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one scenario is held in memory: a scenario is handed over before
    the next scenario is solved."""
    study_dir = _make_study(tmp_path, mode)
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    events: List[Tuple[str, Optional[int]]] = []
    solve_block = SimulationSession._solve_block

    def spy(self, block, scenario_ids, initial_values=None):  # type: ignore[no-untyped-def]
        events.append(("solve", scenario_ids[0]))
        return solve_block(self, block, scenario_ids, initial_values)

    monkeypatch.setattr(SimulationSession, "_solve_block", spy)
    SimulationSession(
        load_study(study_dir),
        optim_config,
        run_id="run",
        on_scenario_done=lambda scenario_id, _table: events.append(
            ("done", scenario_id)
        ),
    ).run()

    assert events.index(("done", 0)) < events.index(("solve", 1))


def test_frontal_hand_over_never_builds_the_full_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frontal results are built per scenario, never as one table holding all
    scenarios."""
    study_dir = _make_study(tmp_path, "frontal")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the full multi-scenario table was built")

    monkeypatch.setattr(SimulationTableBuilder, "build", fail)
    _, streamed = _stream(study_dir)

    assert set(streamed) == {None, 0, 1}


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_empty_scenario_scope_fails_before_solving(tmp_path: Path, mode: str) -> None:
    """Rejected by the optim-config validation, before anything is solved: in
    frontal mode, building the problem for no scenario would fail with another
    error, and sequential/parallel runs would end without output."""
    study_dir = _make_study(tmp_path, mode)
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(
        config_path.read_text().replace("    - 1\n", "    - 1\n  exclude: [0, 1]\n")
    )

    with pytest.raises(ValueError, match="empty scenario list"):
        run_study(study_dir)
    assert not (study_dir / "output").exists()
