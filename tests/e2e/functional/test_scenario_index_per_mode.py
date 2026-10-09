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
- parallel/sequential subproblems: each scenario is solved separately and sizes
  its own ``p_max``, so every row must carry the scenario it was solved for.
  Block ids restart at 0 for each scenario, so without this tag the rows of the
  two scenarios would share their full key.
"""

import shutil
import textwrap
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from gems_craft.study.folder import load_study
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.simulation_table import SimulationTableBuilder
from gems_runner.study.runner import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "13_1"

_CONFIG_TEMPLATE = textwrap.dedent("""\
    time-scope:
      first-time-step: 0
      last-time-step: 0
    solver-options:
      name: highs
      logs: false
      parameters: ""
    scenario-scope:
      include: {scenarios}
""")

_PARALLEL_RESOLUTION = textwrap.dedent("""\
    resolution:
      mode: parallel-subproblems
      block-length: 1
""")

_KEY = ["block", "component", "output", "absolute_time_index", "scenario_index"]


def _two_load_study(tmp_path: Path) -> Path:
    """Copy 13_1 with a load of 300 in scenario 0 and 500 in scenario 1."""
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
    return study_dir


def _run_study(
    tmp_path: Path, resolution: str, scenarios: Tuple[int, ...] = (0, 1)
) -> pd.DataFrame:
    study_dir = _two_load_study(tmp_path)
    (study_dir / "input" / "optim-config.yml").write_text(
        _CONFIG_TEMPLATE.format(scenarios=list(scenarios)) + resolution
    )

    run_study(study_dir)

    output_files = sorted((study_dir / "output").glob("**/simulation_table_*.csv"))
    assert len(output_files) == 1
    return pd.read_csv(output_files[0])


def _rows(df: pd.DataFrame, output: str) -> pd.DataFrame:
    return df[df["output"] == output].sort_values("value")


def test_frontal_shared_rows_have_no_scenario(tmp_path: Path) -> None:
    df = _run_study(tmp_path, resolution="")

    p_max = _rows(df, "p_max")
    objective = _rows(df, "objective-value")
    assert len(p_max) == 1 and p_max["scenario_index"].isna().all()
    assert len(objective) == 1 and objective["scenario_index"].isna().all()


def test_parallel_rows_are_tagged_with_their_scenario(tmp_path: Path) -> None:
    df = _run_study(tmp_path, resolution=_PARALLEL_RESOLUTION)

    assert df["scenario_index"].notna().all()
    assert not df.duplicated(subset=_KEY).any()
    # Each scenario sizes its own capacity: the higher load needs more.
    p_max = _rows(df, "p_max")
    assert list(p_max["scenario_index"]) == [0, 1]
    assert list(p_max["value"]) == pytest.approx([100.0, 300.0])
    objective = _rows(df, "objective-value")
    assert list(objective["scenario_index"]) == [0, 1]
    assert list(objective["value"]) == pytest.approx([50_000.0, 132_000.0])


def test_frontal_single_scenario_rows_are_tagged(tmp_path: Path) -> None:
    df = _run_study(tmp_path, resolution="", scenarios=(1,))

    assert list(df["scenario_index"].unique()) == [1]
    assert list(_rows(df, "p_max")["value"]) == pytest.approx([300.0])


def test_build_without_remap_uses_the_problem_scenarios(tmp_path: Path) -> None:
    """The documented low-level path: build_problem, solve, then build(problem).

    build_problem records the scenario on the problem, so its rows are tagged
    even without scenario_ids_remap.
    """
    problem = build_problem(
        load_study(_two_load_study(tmp_path)), TimeBlock(0, [0]), [0]
    )
    problem.solve(solver_name="highs")
    df = SimulationTableBuilder().build(problem).data

    shared = df[df["output"].isin(["p_max", "objective-value"])]
    assert list(shared["scenario_index"]) == [0, 0]
