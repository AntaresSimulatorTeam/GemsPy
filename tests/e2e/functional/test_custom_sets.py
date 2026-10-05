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

import numpy as np
import pytest
import xarray as xr

from gems_craft.expression.expression import SetIndexNode, SumOverNode, param
from gems_craft.expression.visitor import visit
from gems_craft.study.folder import load_study
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.linearize import VectorizedLinearExprBuilder

STUDY_DIR = Path(__file__).parent / "studies" / "custom_sets"

NAN = np.nan

# Expected solution of the joint LP (checked with scipy), `nan` where a set dimension
# is padded beyond the component's own size (`seg`: 3, `slot`: 2).
EXPECTED = {
    ("global-plant", "gen", "G"): [2, 0],
    ("global-plant", "gen", "G2"): [0, 9],
    ("grid", "x", "P"): [[0, 1], [1, 0], [NAN, NAN]],
    ("grid", "x", "Q"): [[1, NAN], [0, NAN], [0, NAN]],
}


def test_custom_sets_study() -> None:
    """Ragged local sets (`seg` and `slot` together), a global set and an
    `indexed-by` override (G2), all sharing one node and one demand.

    The cyclic ramp `x[seg+1] - x <= 1` of `grid` binds for P and for Q (through
    the wrap); `gen[fuel=0] <= 2` binds for G.
    """
    problem = build_problem(load_study(STUDY_DIR), TimeBlock(1, [0]), [0])
    problem.solve()
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(25.4)

    for (model, var, component), expected in EXPECTED.items():
        solution = problem.get_variable_solution(f"custom_sets.{model}", var)
        np.testing.assert_allclose(
            solution.sel(component=component).values.squeeze(),
            expected,
            equal_nan=True,
            err_msg=f"{component}.{var}",
        )


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
