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

import errno
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_craft.study.parsing import OutputFormat
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
)
from gems_runner.simulation.simulation_table_writer import SimulationTableWriter
from gems_runner.simulation.time_block import TimeBlock
from gems_runner.study.runner import INCOMPLETE_DIR_NAME, run_study

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


def _session(study_dir: Path) -> SimulationSession:
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    return SimulationSession(load_study(study_dir), optim_config, run_id="run")


def _scenario_of(table: SimulationTable) -> Optional[int]:
    """The scenario of a yielded table, or None for the rows shared by all
    scenarios; a yielded table never mixes scenarios."""
    scenarios = table.data["scenario_index"].unique()
    assert len(scenarios) == 1, f"table mixes scenarios {list(scenarios)}"
    return None if pd.isna(scenarios[0]) else int(scenarios[0])


def _stream(study_dir: Path) -> Dict[Optional[int], SimulationTable]:
    """Iterate over the results and collect them by scenario."""
    streamed: Dict[Optional[int], SimulationTable] = {}
    for table in _session(study_dir).iter_scenario_tables():
        scenario_id = _scenario_of(table)
        assert scenario_id not in streamed, "scenario yielded twice"
        streamed[scenario_id] = table
    return streamed


def _write_streamed(
    writer: SimulationTableWriter,
    streamed: Dict[Optional[int], SimulationTable],
    output_dir: Path,
) -> List[Path]:
    """Write each yielded table, which must go to a file of its own."""
    paths: List[Path] = []
    for table in streamed.values():
        written = writer.write(table, output_dir)
        assert len(written) == 1, written
        paths.extend(written)
    return sorted(paths)


def _full_table(study_dir: Path, mode: str) -> SimulationTable:
    """Reference table holding all scenarios, built without
    iter_scenario_tables: in frontal mode, from the solved problem in one go;
    in the other modes, the scenarios are solved separately anyway."""
    session = _session(study_dir)
    if mode != "frontal":
        return session.run()
    time_scope = session.optim_config.time_scope
    block = TimeBlock(
        0, list(range(time_scope.first_time_step, time_scope.last_time_step + 1))
    )
    problem = session._solve_block(block, scenario_ids=session.scenario_ids)
    return SimulationTableBuilder().build(
        problem, scenario_ids_remap=session.scenario_ids, table_id="run"
    )


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_streamed_files_match_splitting_the_full_table(
    tmp_path: Path, mode: str
) -> None:
    """Yielding results one scenario at a time gives byte-identical files to
    solving everything first and splitting the full table, in both formats
    (each study is solved once and written in both formats)."""
    study_dir = _make_study(tmp_path, mode)

    full_table = _full_table(study_dir, mode)
    streamed = _stream(study_dir)

    for output_format in OutputFormat:
        writer = SimulationTableWriter(output_format)
        out_dir = tmp_path / output_format.value
        split_paths = sorted(writer.write(full_table, out_dir / "split"))
        streamed_paths = _write_streamed(writer, streamed, out_dir / "streamed")

        assert split_paths, output_format
        assert [p.name for p in split_paths] == [p.name for p in streamed_paths]
        for split_path, streamed_path in zip(split_paths, streamed_paths, strict=True):
            assert (
                split_path.read_bytes() == streamed_path.read_bytes()
            ), split_path.name


def test_frontal_hands_over_common_rows_once_and_each_scenario_once(
    tmp_path: Path,
) -> None:
    streamed = _stream(_make_study(tmp_path, "frontal"))

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

    full_table = _full_table(study_dir, "frontal")
    split_paths = sorted(writer.write(full_table, tmp_path / "split"))
    streamed = _stream(study_dir)
    streamed_paths = _write_streamed(writer, streamed, tmp_path / "streamed")

    assert _suffixes(streamed_paths) == ["scenario-0", "scenario-2", "scenario-common"]
    assert [p.name for p in split_paths] == [p.name for p in streamed_paths]
    for split_path, streamed_path in zip(split_paths, streamed_paths, strict=True):
        assert split_path.read_bytes() == streamed_path.read_bytes(), split_path.name
    scenario_2 = pd.read_parquet(streamed_paths[1])
    assert (scenario_2["scenario_index"] == 2).all()


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_each_scenario_is_yielded_before_the_next_is_solved(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one scenario is held in memory: a scenario is yielded before the
    next scenario is solved."""
    study_dir = _make_study(tmp_path, mode)
    events: List[Tuple[str, Optional[int]]] = []
    solve_block = SimulationSession._solve_block

    def spy(self, block, scenario_ids, initial_values=None):  # type: ignore[no-untyped-def]
        events.append(("solve", scenario_ids[0]))
        return solve_block(self, block, scenario_ids, initial_values)

    monkeypatch.setattr(SimulationSession, "_solve_block", spy)
    for table in _session(study_dir).iter_scenario_tables():
        events.append(("done", _scenario_of(table)))

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
    streamed = _stream(study_dir)

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


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_run_returns_the_merge_of_the_iterated_results(
    tmp_path: Path, mode: str
) -> None:
    study_dir = _make_study(tmp_path, mode)

    returned = _session(study_dir).run()
    iterated = [t.data for t in _session(study_dir).iter_scenario_tables()]

    assert returned.table_id == "run"
    pd.testing.assert_frame_equal(returned.data, pd.concat(iterated, ignore_index=True))


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_run_holds_the_same_rows_as_the_full_table(tmp_path: Path, mode: str) -> None:
    """run() merges the per-scenario results: same rows as the table built in
    one go from the solution, possibly in another order."""
    study_dir = _make_study(tmp_path, mode)
    columns = ["block", "component", "output", "absolute_time_index", "scenario_index"]

    def rows(table: SimulationTable) -> pd.DataFrame:
        return table.data.sort_values(columns, na_position="first").reset_index(
            drop=True
        )

    pd.testing.assert_frame_equal(
        rows(_session(study_dir).run()), rows(_full_table(study_dir, mode))
    )


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_nothing_is_solved_before_the_iteration_starts(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_dir = _make_study(tmp_path, mode)
    solved: List[int] = []
    solve_block = SimulationSession._solve_block

    def spy(self, block, scenario_ids, initial_values=None):  # type: ignore[no-untyped-def]
        solved.append(scenario_ids[0])
        return solve_block(self, block, scenario_ids, initial_values)

    monkeypatch.setattr(SimulationSession, "_solve_block", spy)
    tables = _session(study_dir).iter_scenario_tables()
    assert solved == []

    first = next(tables)
    assert solved
    assert _scenario_of(first) in (None, 0)


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_invalid_config_fails_when_iteration_is_requested(
    tmp_path: Path, mode: str
) -> None:
    """The optim-config is validated when iter_scenario_tables is called,
    not at the first next()."""
    study_dir = _make_study(tmp_path, mode)
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(
        config_path.read_text().replace("    - 1\n", "    - 1\n  exclude: [0, 1]\n")
    )

    with pytest.raises(ValueError, match="empty scenario list"):
        _session(study_dir).iter_scenario_tables()


# ---------------------------------------------------------------------------
# Run folder: output/<run_id>/ only holds completed runs
# ---------------------------------------------------------------------------


class _FixedTime:
    """Stands for datetime in the runner: every run starts in the same minute."""

    @staticmethod
    def now() -> datetime:
        return datetime(2026, 10, 8, 10, 12, 30)


def _output_entries(study_dir: Path) -> List[str]:
    output = study_dir / "output"
    return sorted(p.name for p in output.iterdir()) if output.exists() else []


def _files(folder: Path) -> Dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


@pytest.mark.parametrize("mode", ["frontal", "sequential", "parallel"])
def test_completed_run_is_moved_to_its_run_folder(tmp_path: Path, mode: str) -> None:
    study_dir = _make_study(tmp_path, mode)
    run_study(study_dir)

    (run_id,) = _output_entries(study_dir)
    assert run_id != INCOMPLETE_DIR_NAME
    files = sorted(_files(study_dir / "output" / run_id))
    assert files[:2] == [
        f"simulation_table_{run_id}_scenario-0.csv",
        f"simulation_table_{run_id}_scenario-1.csv",
    ]


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_failed_run_leaves_no_output(tmp_path: Path, mode: str) -> None:
    """Scenario 2 has no column in the data series: the run fails after
    scenarios 0 and 1 were written. Nothing is left in output/, and the error
    names the failing scenario and block."""
    study_dir = _make_study(tmp_path, mode)
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(
        config_path.read_text().replace("    - 1\n", "    - 1\n    - 2\n")
    )

    with pytest.raises(IndexError) as raised:
        run_study(study_dir)

    assert not (study_dir / "output").exists()
    assert any(
        note.startswith("While solving scenario 2, block 0")
        for note in getattr(raised.value, "__notes__", [])
    )


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk-full", "ctrl-c"],
)
def test_run_interrupted_while_writing_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    """Frontal mode: the scenario-common file and scenario 0 are written, then
    writing scenario 1 fails."""
    study_dir = _make_study(tmp_path, "frontal")
    write = SimulationTableWriter.write
    calls: List[int] = []

    def failing_write(  # type: ignore[no-untyped-def]
        self, table: SimulationTable, output_dir: Path
    ) -> List[Path]:
        calls.append(1)
        if len(calls) == 3:
            raise error
        return write(self, table, output_dir)

    monkeypatch.setattr(SimulationTableWriter, "write", failing_write)
    with pytest.raises(type(error)):
        run_study(study_dir)

    assert len(calls) == 3
    assert not (study_dir / "output").exists()


def test_failed_run_keeps_the_other_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_dir = _make_study(tmp_path, "sequential")
    run_study(study_dir)
    (completed,) = _output_entries(study_dir)
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(
        config_path.read_text().replace("    - 1\n", "    - 1\n    - 2\n")
    )

    with pytest.raises(IndexError):
        run_study(study_dir)

    assert _output_entries(study_dir) == [completed]


def test_runs_started_in_the_same_minute_get_their_own_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second run gets a -2 suffix, in its folder and file names, and the
    first run's files are left as they were."""
    monkeypatch.setattr("gems_runner.study.runner.datetime", _FixedTime)
    study_dir = _make_study(tmp_path, "frontal")
    run_study(study_dir)
    first_run = _files(study_dir / "output" / "20261008T1012")

    run_study(study_dir)

    assert _output_entries(study_dir) == ["20261008T1012", "20261008T1012-2"]
    assert _files(study_dir / "output" / "20261008T1012") == first_run
    assert sorted(_files(study_dir / "output" / "20261008T1012-2")) == [
        "simulation_table_20261008T1012-2_scenario-0.csv",
        "simulation_table_20261008T1012-2_scenario-1.csv",
        "simulation_table_20261008T1012-2_scenario-common.csv",
    ]


def test_folder_left_by_a_killed_run_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run leaves its folder in output/incomplete/: a new run started
    in the same minute neither reuses nor removes it."""
    monkeypatch.setattr("gems_runner.study.runner.datetime", _FixedTime)
    study_dir = _make_study(tmp_path, "frontal")
    leftover = study_dir / "output" / INCOMPLETE_DIR_NAME / "20261008T1012"
    leftover.mkdir(parents=True)
    (leftover / "partial.csv").write_text("partial")

    run_study(study_dir)

    assert _output_entries(study_dir) == ["20261008T1012-2", INCOMPLETE_DIR_NAME]
    assert _files(leftover) == {"partial.csv": b"partial"}


def test_files_written_by_the_session_are_moved_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Benders mode writes its own files (structure.txt...) into the session's
    output folder, and no simulation table: they end up in output/<run_id>/."""
    study_dir = _make_study(tmp_path, "frontal")

    def benders_like(self: SimulationSession) -> Iterator[SimulationTable]:
        assert self.output_dir is not None
        (self.output_dir / "structure.txt").write_text("structure")
        yield from ()

    monkeypatch.setattr(SimulationSession, "iter_scenario_tables", benders_like)
    run_study(study_dir)

    (run_id,) = _output_entries(study_dir)
    assert _files(study_dir / "output" / run_id) == {"structure.txt": b"structure"}
