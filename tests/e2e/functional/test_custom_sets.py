# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gems_craft.study.folder import load_study
from gems_runner.simulation import SimulationTable, TimeBlock, build_problem
from gems_runner.simulation.simulation_table import SimulationTableBuilder

STUDY_DIR = Path(__file__).parent / "studies" / "custom_sets"

SET_IDS = {
    ("G", "gen"): "fuel",
    ("G2", "gen"): "fuel",
    ("P", "x"): "seg|slot",
    ("Q", "x"): "seg|slot",
}


def test_custom_sets_study(tmp_path: Path) -> None:
    """Ragged local sets (`seg` and `slot` together), a global set and an
    `indexed-by` override (G2), all sharing one node and one demand.

    The cyclic ramp `x[seg+1] - x <= 1` of `grid` binds for P and for Q (through
    the wrap); `gen[fuel=0] <= 2` binds for G.

    Set-indexed outputs carry pipe-joined `set_id` / `set_index` (element names,
    sets sorted by id); padded positions of ragged sets are dropped.
    """
    problem = build_problem(load_study(STUDY_DIR), TimeBlock(1, [0]), [0])
    problem.solve()
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(25.4)

    table = SimulationTableBuilder().build(problem)
    df = table.data

    def rows(component: str, output: str) -> dict:  # type: ignore[type-arg]
        sub = df[(df["component"] == component) & (df["output"] == output)]
        assert set(sub["set_id"]) == {SET_IDS[(component, output)]}
        return dict(zip(sub["set_index"], sub["value"]))

    assert rows("G", "gen") == {"coal": pytest.approx(2), "gas": pytest.approx(0)}
    assert rows("G2", "gen") == {"coal": pytest.approx(0), "gas": pytest.approx(9)}
    assert rows("P", "x") == {
        "s0|f0": pytest.approx(0),
        "s0|f1": pytest.approx(1),
        "s1|f0": pytest.approx(1),
        "s1|f1": pytest.approx(0),
    }
    assert rows("Q", "x") == {
        "s0|f0": pytest.approx(1),
        "s1|f0": pytest.approx(0),
        "s2|f0": pytest.approx(0),
    }

    obj = df[df["output"] == "objective-value"]
    assert obj["set_id"].isna().all() and obj["set_index"].isna().all()

    view = table.component("P").output("x")
    assert view.data.columns.names == ["scenario_index", "set_index"]
    assert view.value(time_index=0, scenario_index=0)["s0|f1"] == 1.0

    stored = SimulationTable.open(table.save(tmp_path / "table"))
    pd.testing.assert_frame_equal(stored.to_pandas(load_all=True), df)


def test_positive_lower_bound_on_ragged_sets(tmp_path: Path) -> None:
    """A lower bound above the padded upper bound (0) is not a bound violation."""
    study_dir = Path(shutil.copytree(STUDY_DIR, tmp_path / "study"))
    library = study_dir / "input" / "model-libraries" / "custom_sets.yml"
    text, count = re.subn(
        r"lower-bound: 0(\r?\n\s+upper-bound: cap)",
        r"lower-bound: 0.5\1",
        library.read_text(),
    )
    assert count == 1
    library.write_text(text)

    problem = build_problem(load_study(study_dir), TimeBlock(1, [0]), [0])
    problem.solve()
    assert problem.termination_condition == "optimal"
    x = problem.get_variable_solution("custom_sets.grid", "x").values
    assert np.all(x[~np.isnan(x)] >= 0.5 - 1e-9)
