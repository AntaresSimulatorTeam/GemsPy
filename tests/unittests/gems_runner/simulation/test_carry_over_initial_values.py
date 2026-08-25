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

"""
Tests the `initial_values` contract of `build_problem`, i.e. how a carry-over
window from the previous block is turned into constraints of the next one:

- only time-dependent variables are pinned — a time-independent variable
  (`structure.time = False`) is left free in every block;
- a value with no `time` dimension is rejected outright.

The pinned *window length*, which `carry-over-length` controls through the
session, is covered end to end in
`tests/e2e/functional/test_sequential_carry_over_length.py`.
"""

import pytest
import xarray as xr

from gems_craft.expression.expression import literal, param, var
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model import Constraint, float_parameter, float_variable, model
from gems_craft.study import ConstantData, DataBase, Study, System, create_component
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.optimization import _validate_initial_values

CONSTANT = IndexingStructure(False, False)


def _one_time_dependent_one_constant_study() -> Study:
    """A single-component study whose model has one time-dependent variable
    (`gen`) and one time-independent one (`cap`)."""
    plant = model(
        id="PLANT",
        parameters=[float_parameter("cost", CONSTANT)],
        variables=[
            float_variable("gen", lower_bound=literal(0), upper_bound=literal(10)),
            float_variable(
                "cap",
                lower_bound=literal(0),
                upper_bound=literal(10),
                structure=CONSTANT,
            ),
        ],
        constraints=[
            Constraint(name="Max generation", expression=var("gen") <= var("cap"))
        ],
        objective_contributions={
            "operational": (param("cost") * var("gen")).time_sum().expec()
        },
    )
    database = DataBase()
    database.add_data("P", "cost", ConstantData(1))
    system = System("carry_over_contract")
    system.add_component(create_component(model=plant, id="P"))
    return Study(system, database)


def _time_da(values: list[float]) -> xr.DataArray:
    """Carry-over array in the shape `_extract_carry_over` produces: a `time`
    dimension indexed 0..k-1."""
    return xr.DataArray(
        values, dims=["time"], coords={"time": list(range(len(values)))}
    )


def test_carry_over_skips_time_independent_variables() -> None:
    """Only time-dependent variables are pinned.  A time-independent variable
    (`structure.time = False`) is left free in every block even when the caller
    passes a value for it, so consecutive blocks size it independently."""
    problem = build_problem(
        _one_time_dependent_one_constant_study(),
        TimeBlock(1, [0, 1, 2]),
        [0],
        initial_values={
            ("PLANT", "gen"): _time_da([2.0, 3.0]),
            ("PLANT", "cap"): _time_da([7.0]),
        },
    )

    constraint_names = set(problem.linopy_model.constraints)
    assert "carry_over__PLANT__gen" in constraint_names
    assert "carry_over__PLANT__cap" not in constraint_names


def test_initial_values_without_time_dim_rejected() -> None:
    """A value with no `time` dimension — the shape carried over before
    multi-timestep stitching existed — is rejected outright rather than
    silently reinterpreted as a single-timestep pin."""
    with pytest.raises(ValueError, match="must carry a 'time' dimension"):
        build_problem(
            _one_time_dependent_one_constant_study(),
            TimeBlock(1, [0, 1, 2]),
            [0],
            initial_values={("PLANT", "gen"): xr.DataArray(2.0)},
        )

    # The check is a precondition on the argument, so it does not need a study:
    with pytest.raises(ValueError, match="must carry a 'time' dimension"):
        _validate_initial_values({("PLANT", "gen"): xr.DataArray(2.0)})

    assert _validate_initial_values(None) == {}
