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

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from gems_craft.expression.expression import SetIndexNode, SumOverNode, param
from gems_craft.expression.visitor import visit
from gems_craft.study.folder import load_study
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.linearize import VectorizedLinearExprBuilder
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

    csv = pd.read_csv(table.to_csv(tmp_path))
    assert {"set_id", "set_index"} <= set(csv.columns)


def test_set_operators_on_ragged_arrays() -> None:
    """`X[fuel+1]`, `X[fuel=k]` and `sum_over` on sets of sizes 2 and 3."""
    comps = ["c0", "c1"]
    price = xr.DataArray(
        [[10.0, 11.0, 0.0], [20.0, 21.0, 22.0]],  # padded with 0
        dims=["component", "fuel"],
        coords={"component": comps, "fuel": [0, 1, 2]},
    )
    builder = VectorizedLinearExprBuilder(
        model_id="m",
        linopy_vars={},
        param_arrays={
            ("m", "price"): price,
            ("m", "one"): xr.DataArray(1.0),
            ("m", "two"): xr.DataArray(2.0),
        },
        port_arrays={},
        block_length=1,
        set_sizes={
            "fuel": xr.DataArray(
                [2, 3], dims=["component"], coords={"component": comps}
            )
        },
    )

    shifted = visit(
        SetIndexNode(param("price"), "fuel", relative_shift=param("one")), builder
    )
    assert shifted.sel(component="c0").values[:2].tolist() == [11.0, 10.0]  # wraps at 2
    assert shifted.sel(component="c1").values.tolist() == [
        21.0,
        22.0,
        20.0,
    ]  # wraps at 3

    picked = visit(SetIndexNode(param("price"), "fuel", position=param("one")), builder)
    assert picked.values.tolist() == [11.0, 21.0]
    with pytest.raises(ValueError, match="out of range"):  # c0 only has 2 elements
        visit(SetIndexNode(param("price"), "fuel", position=param("two")), builder)

    summed = visit(SumOverNode(param("price"), "fuel"), builder)
    assert summed.values.tolist() == [21.0, 63.0]
