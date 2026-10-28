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

"""E2E test: a session given an output_dir stores its results on disk, block
by block as they are solved, and returns the stored table."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionConfig,
    ResolutionMode,
    ScenarioScopeConfig,
    TimeScopeConfig,
)
from gems_craft.study.folder import load_study
from gems_craft.study.parsing import OutputFormat
from gems_runner.session import session as session_module
from gems_runner.session.session import SIMULATION_TABLE_FOLDER, SimulationSession
from gems_runner.simulation.simulation_table import (
    SIMULATION_TABLE_DTYPES,
    SimulationTable,
    SimulationTableBuilder,
)
from gems_runner.simulation.table_folder import MANIFEST_FILE, _TableWriter
from gems_runner.study.runner import run_study
from tests.e2e.functional.simulation_table_output import assert_same_tables

_STUDY_13_1 = Path(__file__).parent / "studies" / "13_1"


def _config(
    mode: ResolutionMode, scenarios: List[int], **resolution: int
) -> OptimConfig:
    return OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=0, last_time_step=3),
        scenario_scope=ScenarioScopeConfig(include=scenarios),
        resolution=ResolutionConfig(mode=mode, **resolution),
    )


_CONFIGS = {
    "frontal": _config(ResolutionMode.FRONTAL, [0, 1]),
    "frontal-1": _config(ResolutionMode.FRONTAL, [1]),
    "sequential-overlap": _config(
        ResolutionMode.SEQUENTIAL_SUBPROBLEMS, [0, 1], block_length=2, block_overlap=1
    ),
    "parallel": _config(ResolutionMode.PARALLEL_SUBPROBLEMS, [0, 1], block_length=2),
}


def _session(config: Any, **kwargs: Any) -> SimulationSession:
    return SimulationSession(load_study(_STUDY_13_1), config, run_id="run", **kwargs)


@pytest.mark.parametrize("fmt", [OutputFormat.PARQUET, OutputFormat.CSV])
@pytest.mark.parametrize("mode", list(_CONFIGS))
def test_stored_table_matches_the_table_in_memory(
    tmp_path: Path, mode: str, fmt: OutputFormat
) -> None:
    memory = _session(_CONFIGS[mode]).run()

    stored = _session(_CONFIGS[mode], output_dir=tmp_path, output_format=fmt).run()

    assert stored.path == tmp_path / SIMULATION_TABLE_FOLDER
    assert stored.scenarios == _CONFIGS[mode].scenario_scope.scenario_ids
    assert_same_tables(memory, stored)


def test_a_folder_holding_a_table_is_refused_before_anything_is_solved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(_CONFIGS["parallel"], output_dir=tmp_path).run()
    solved: List[Any] = []
    monkeypatch.setattr(
        SimulationSession, "_solve_block", lambda self, *a: solved.append(a)
    )

    with pytest.raises(FileExistsError, match="is not empty"):
        _session(_CONFIGS["parallel"], output_dir=tmp_path).run()
    assert solved == []


_SHARED_ONLY_LIBRARY = """\
library:
  id: lib
  models:
    - id: inv
      parameters:
        - id: c
          scenario-dependent: false
          time-dependent: false
      variables:
        - id: x
          scenario-dependent: false
          time-dependent: false
          lower-bound: 1
          upper-bound: 10
        - id: y
          scenario-dependent: false
          lower-bound: 2
          upper-bound: 10
      extra-outputs:
        - id: cx
          expression: c * x
      objective-contributions:
        - id: obj
          expression: c * x + sum(c * y)
"""

_SHARED_ONLY_SYSTEM = """\
system:
  id: shared
  components:
    - id: g
      model: lib.inv
      parameters:
        - id: c
          scenario-dependent: false
          time-dependent: false
          value: 3
"""


@pytest.mark.parametrize("fmt", [OutputFormat.PARQUET, OutputFormat.CSV])
def test_frontal_outputs_shared_by_every_scenario_go_to_the_common_file(
    tmp_path: Path, fmt: OutputFormat
) -> None:
    study = tmp_path / "study"
    (study / "input" / "model-libraries").mkdir(parents=True)
    (study / "input" / "model-libraries" / "lib.yml").write_text(_SHARED_ONLY_LIBRARY)
    (study / "input" / "system.yml").write_text(_SHARED_ONLY_SYSTEM)
    config = _config(ResolutionMode.FRONTAL, [0, 1])

    memory = SimulationSession(load_study(study), config, run_id="run").run()
    stored = SimulationSession(
        load_study(study),
        config,
        run_id="run",
        output_dir=tmp_path / "out",
        output_format=fmt,
    ).run()

    assert_same_tables(memory, stored)
    parts = [(s, len(rows)) for s, rows in stored.iter_scenario_frames()]
    assert parts[0][0] is None  # every row is in the common file
    assert parts[1:] == [(0, 0), (1, 0)]
    manifest = json.loads((stored.path / MANIFEST_FILE).read_text())
    assert manifest["scenarios"] == [0, 1]
    assert manifest["has_common"] is True


@pytest.mark.parametrize("mode", ["sequential-overlap", "parallel"])
def test_each_block_is_written_before_the_next_is_solved(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: List[Tuple[str, int]] = []
    solve, append = SimulationSession._solve_block, _TableWriter.append

    def spy_solve(self: Any, block: Any, scenario_ids: List[int], *a: Any) -> Any:
        events.append(("solve", scenario_ids[0]))
        return solve(self, block, scenario_ids, *a)

    def spy_append(self: _TableWriter, df: Any) -> None:
        events.append(("write", int(df["scenario_index"].iloc[0])))
        append(self, df)

    monkeypatch.setattr(SimulationSession, "_solve_block", spy_solve)
    monkeypatch.setattr(_TableWriter, "append", spy_append)

    _session(_CONFIGS[mode], output_dir=tmp_path).run()

    # solve, write, solve, write, ...: no block waits for the next to be solved
    assert [k for k, _ in events] == ["solve", "write"] * (len(events) // 2)
    assert [s for k, s in events if k == "solve"] == [
        s for k, s in events if k == "write"
    ]
    assert len(events) > 4  # several blocks per scenario


def test_frontal_never_builds_the_full_table_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Any] = []
    build_frame = SimulationTableBuilder._build_frame

    def spy(self: Any, problem: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("scenario_ids_remap"))
        return build_frame(self, problem, *args, **kwargs)

    monkeypatch.setattr(SimulationTableBuilder, "_build_frame", spy)

    _session(_CONFIGS["frontal"], output_dir=tmp_path).run()
    assert calls == []  # rows built one scenario at a time

    _session(_CONFIGS["frontal"]).run()
    assert calls == [[0, 1]]  # in memory: one table, as before


def test_an_interrupted_run_leaves_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve = SimulationSession._solve_block

    def failing(self: Any, block: Any, scenario_ids: List[int], *a: Any) -> Any:
        if scenario_ids == [1]:
            raise RuntimeError("solver crashed")
        return solve(self, block, scenario_ids, *a)

    monkeypatch.setattr(SimulationSession, "_solve_block", failing)

    with pytest.raises(RuntimeError, match="solver crashed"):
        _session(_CONFIGS["parallel"], output_dir=tmp_path).run()

    folder = tmp_path / SIMULATION_TABLE_FOLDER
    assert sorted(f.name for f in folder.iterdir()) == ["scenario-0.parquet"]
    with pytest.raises(FileNotFoundError, match="has no manifest.json"):
        SimulationTable.open(folder)


@pytest.mark.parametrize("fmt", [OutputFormat.PARQUET, OutputFormat.CSV])
@pytest.mark.parametrize("error", [RuntimeError("solver crashed"), KeyboardInterrupt()])
def test_a_run_interrupted_within_a_scenario_keeps_only_complete_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: OutputFormat, error: Any
) -> None:
    solve, solved = SimulationSession._solve_block, []

    def fail_on_the_second_block_of_scenario_1(
        self: Any, block: Any, scenario_ids: List[int], *a: Any
    ) -> Any:
        solved.append(scenario_ids[0])
        if solved.count(1) == 2:  # scenario 1's file is open, one block in
            raise error
        return solve(self, block, scenario_ids, *a)

    monkeypatch.setattr(
        SimulationSession, "_solve_block", fail_on_the_second_block_of_scenario_1
    )

    with pytest.raises(type(error)):
        _session(_CONFIGS["parallel"], output_dir=tmp_path, output_format=fmt).run()

    folder = tmp_path / SIMULATION_TABLE_FOLDER
    assert sorted(f.name for f in folder.iterdir()) == [f"scenario-0.{fmt.value}"]
    with pytest.raises(FileNotFoundError, match="has no manifest.json"):
        SimulationTable.open(folder)


def test_benders_writes_no_table_and_returns_an_empty_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Benders mode does not run here: only what run() returns is checked.
    monkeypatch.setattr(SimulationSession, "_run_benders", lambda self: None)
    monkeypatch.setattr(session_module, "validate_optim_config", lambda *a: None)

    table = _session(
        _config(ResolutionMode.BENDERS_DECOMPOSITION, [0, 1]), output_dir=tmp_path
    ).run()

    assert not (tmp_path / SIMULATION_TABLE_FOLDER).exists()
    assert table.data.empty
    assert table.data.dtypes.to_dict() == SIMULATION_TABLE_DTYPES
    assert table.scenarios == [0, 1]
    assert table.objective_values().empty


@pytest.mark.parametrize(
    "fmt, ext",
    [
        pytest.param(None, "parquet", id="default"),
        pytest.param(OutputFormat.CSV, "csv", id="csv"),
    ],
)
def test_run_study_writes_a_table_folder(tmp_path: Path, fmt: Any, ext: str) -> None:
    study_dir = tmp_path / "13_1"
    shutil.copytree(_STUDY_13_1, study_dir)

    run_study(study_dir) if fmt is None else run_study(study_dir, output_format=fmt)

    (run_dir,) = (study_dir / "output").iterdir()
    folder = run_dir / SIMULATION_TABLE_FOLDER
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    assert manifest["format"] == ext
    assert manifest["run_id"] == run_dir.name
    assert (folder / f"scenario-{manifest['scenarios'][0]}.{ext}").is_file()
    assert not SimulationTable.open(folder).objective_values().empty


@pytest.mark.parametrize("with_output_dir", [False, True])
def test_sequential_mode_with_a_single_time_step_raises(
    tmp_path: Path, with_output_dir: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sequential mode solves no block on a single time step: raise before
    # anything is solved or written, rather than return an empty table.
    solved: List[Any] = []
    monkeypatch.setattr(
        SimulationSession, "_solve_block", lambda self, *a: solved.append(a)
    )
    config = OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=0, last_time_step=0),
        scenario_scope=ScenarioScopeConfig(include=[0]),
        resolution=ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS, block_length=2
        ),
    )
    kwargs = {"output_dir": tmp_path} if with_output_dir else {}
    with pytest.raises(ValueError, match=r"single time step \(0\.\.0\)"):
        _session(config, **kwargs).run()
    assert solved == []
    assert not (tmp_path / SIMULATION_TABLE_FOLDER).exists()


def test_a_relative_output_dir_is_opened_where_it_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    solve = SimulationSession._solve_block

    def solve_then_chdir(self: Any, *args: Any) -> Any:
        problem = solve(self, *args)
        os.chdir(tmp_path / "b")  # the working directory changes mid-run
        return problem

    monkeypatch.setattr(SimulationSession, "_solve_block", solve_then_chdir)

    stored = _session(_CONFIGS["parallel"], output_dir=Path("run")).run()

    assert stored.path == (tmp_path / "a" / "run" / SIMULATION_TABLE_FOLDER).resolve()
    assert not stored.objective_values().empty
