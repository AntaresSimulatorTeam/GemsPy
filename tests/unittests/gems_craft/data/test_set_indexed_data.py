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
"""Unit tests for set-indexed (tidy CSV) data series."""

import io
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pytest

from gems_craft.study.data import (
    ConstantData,
    SetIndexedSeriesData,
    TimeSeriesData,
    dataframe_to_set_indexed_series,
    load_tidy_series_from_file,
)
from gems_craft.study.parsing import parse_yaml_system
from gems_craft.study.resolve_components import build_data_base

FUELS: Dict[str, List[Union[str, int]]] = {"fuel": ["gas", "coal"]}


def _df(csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv), dtype=str, skipinitialspace=True)


TIME_FUEL_CSV = """fuel,time,value
gas,0,1
gas,1,2
coal,0,10
coal,1,20
"""


def test_time_and_set_pivot() -> None:
    data = dataframe_to_set_indexed_series(_df(TIME_FUEL_CSV), True, False, FUELS)
    assert data.dims == ("time", "fuel")
    assert data.coords == {"fuel": ("gas", "coal")}
    np.testing.assert_array_equal(data.values, [[1, 10], [2, 20]])


@pytest.mark.parametrize(
    "csv",
    [
        "value,time,fuel\n1,0,gas\n2,1,gas\n10,0,coal\n20,1,coal\n",  # columns
        "fuel,time,value\ncoal,1,20\ngas,0,1\ncoal,0,10\ngas,1,2\n",  # rows
    ],
    ids=["column order", "row order"],
)
def test_column_and_row_order_are_irrelevant(csv: str) -> None:
    data = dataframe_to_set_indexed_series(_df(csv), True, False, FUELS)
    np.testing.assert_array_equal(data.values, [[1, 10], [2, 20]])


def test_multi_set_and_int_elements() -> None:
    sets: Dict[str, List[Union[str, int]]] = {"seg": [0, 1], "fuel": ["gas", "coal"]}
    csv = "seg,fuel,value\n" + "".join(
        f"{s},{f},{10 * s + i}\n" for s in (0, 1) for i, f in enumerate(["gas", "coal"])
    )
    data = dataframe_to_set_indexed_series(_df(csv), False, False, sets)
    assert data.dims == ("seg", "fuel")
    np.testing.assert_array_equal(data.values, [[0, 1], [10, 11]])


def test_all_dimensions() -> None:
    csv = "time,scenario,fuel,value\n" + "".join(
        f"{t},{s},{f},{100 * t + 10 * s + i}\n"
        for t in (0, 1)
        for s in (0, 1, 2)
        for i, f in enumerate(["gas", "coal"])
    )
    data = dataframe_to_set_indexed_series(_df(csv), True, True, FUELS)
    assert data.values.shape == (2, 3, 2)
    assert data.get_value([1], np.array([2])).shape == (1, 1, 2)
    assert data.get_value([1], np.array([2]))[0, 0, 1] == 121


def test_overridden_out_dimension_broadcasts() -> None:  # (b)
    data = dataframe_to_set_indexed_series(
        _df("fuel,value\ngas,1\ncoal,2\n"), False, False, FUELS
    )
    out = data.get_value([0, 1, 2], np.array([0, 1]))
    assert out.shape == (3, 2, 2)
    assert (out[:, :, 1] == 2).all()
    assert data.get_value(None, None).shape == (2,)


def test_constant_values_behave_like_varying_data() -> None:  # (d)
    csv = "fuel,time,value\ngas,0,5\ngas,1,5\ncoal,0,5\ncoal,1,5\n"
    data = dataframe_to_set_indexed_series(_df(csv), True, False, FUELS)
    assert data.dims == ("time", "fuel")
    assert data.get_value([0, 1], None).shape == (2, 2)


@pytest.mark.parametrize(
    "csv, message, time_dependent",
    [
        (TIME_FUEL_CSV + "gas,1,3\n", "duplicate or missing", True),
        (
            "fuel,time,value\ngas,0,1\ngas,1,2\ncoal,0,10\n",
            "duplicate or missing",
            True,
        ),
        ("fuel,time,value\ngas,0,1\ncoal,0,2\noil,0,3\n", "do not match", True),
        ("fuel,time,value\ngas,0,1\ngas,2,1\ncoal,0,1\ncoal,2,1\n", "0..n-1", True),
        ("fuel,time,value\ngas,a,1\ncoal,0,1\n", "integers", True),
        ("fuel,time,value\ngas,0,x\ncoal,0,1\n", "numbers", True),
        ("fuel,time,value\ngas,0,\ncoal,0,1\n", "empty cells", True),
        (
            "fuel,time,value,extra\ngas,0,1,1\ncoal,0,1,1\n",
            "unexpected \\['extra'\\]",
            True,
        ),
        ("fuel,value\ngas,1\ncoal,2\n", "missing \\['time'\\]", True),
        (TIME_FUEL_CSV, "unexpected \\['time'\\]", False),
    ],
)
def test_invalid_series(csv: str, message: str, time_dependent: bool) -> None:
    with pytest.raises(ValueError, match=message):
        dataframe_to_set_indexed_series(_df(csv), time_dependent, False, FUELS)


def test_check_requirement() -> None:
    data = dataframe_to_set_indexed_series(_df(TIME_FUEL_CSV), True, False, FUELS)
    assert data.check_requirement(True, True, frozenset({"fuel", "other"}))
    assert not data.check_requirement(False, True, frozenset({"fuel"}))
    assert not data.check_requirement(True, True, frozenset())
    # Legacy structures ignore the sets argument.
    assert ConstantData(1.0).check_requirement(False, False, frozenset({"fuel"}))
    assert TimeSeriesData(pd.Series([1.0])).check_requirement(
        True, False, frozenset({"fuel"})
    )


def test_get_value_requires_declared_indices() -> None:
    data = dataframe_to_set_indexed_series(_df(TIME_FUEL_CSV), True, False, FUELS)
    with pytest.raises(KeyError):
        data.get_value(None, None)


def test_load_tidy_series_from_file(tmp_path: Path) -> None:
    (tmp_path / "s.csv").write_text(TIME_FUEL_CSV)
    assert load_tidy_series_from_file("s", tmp_path).shape == (4, 3)
    with pytest.raises(FileNotFoundError):
        load_tidy_series_from_file("missing", tmp_path)


SYSTEM_YAML = """
system:
  sets:
    - id: fuel
      elements: [gas, coal]
  components:
    - id: C
      model: lib.m
      sets:
        - id: seg
          elements: "0..1"
      parameters:
        - id: price
          time-dependent: true
          indexed-by: [fuel, seg]
          value: price
        - id: cap
          value: 3
"""


def test_build_data_base_with_local_and_global_sets(tmp_path: Path) -> None:
    csv = "time,fuel,seg,value\n" + "".join(
        f"{t},{f},{s},{t + 10 * s + (5 if f == 'coal' else 0)}\n"
        for t in (0, 1)
        for f in ("gas", "coal")
        for s in (0, 1)
    )
    (tmp_path / "price.csv").write_text(csv)
    db = build_data_base(parse_yaml_system(io.StringIO(SYSTEM_YAML)), tmp_path)
    data = db.get_data("C", "price")
    assert isinstance(data, SetIndexedSeriesData)
    assert data.dims == ("time", "fuel", "seg")
    assert data.values[1, 1, 1] == 16
    assert isinstance(db.get_data("C", "cap"), ConstantData)


def test_build_data_base_unknown_set(tmp_path: Path) -> None:
    yaml = SYSTEM_YAML.replace("[fuel, seg]", "[fuel, nope]")
    with pytest.raises(ValueError, match="'nope' is not instantiated"):
        build_data_base(parse_yaml_system(io.StringIO(yaml)), tmp_path)


def test_build_data_base_constant_for_indexed_parameter(tmp_path: Path) -> None:
    yaml = SYSTEM_YAML.replace("value: price", "value: 2.0")
    with pytest.raises(ValueError, match="series name is expected"):
        build_data_base(parse_yaml_system(io.StringIO(yaml)), tmp_path)


INLINE_YAML = """
system:
  sets:
    - id: fuel
      elements: [gas, coal]
  components:
    - id: C
      model: lib.m
      sets:
        - id: seg
          elements: [1, 2]
      parameters:
        - id: cost
          indexed-by: [fuel]
          value: {gas: 45, coal: 30}
        - id: cap
          indexed-by: [fuel, seg]
          value: {gas: {1: 3, 2: 4}, coal: {1: 5, 2: 6.5}}
"""


def test_inline_values() -> None:
    db = build_data_base(parse_yaml_system(io.StringIO(INLINE_YAML)), None)
    cost, cap = db.get_data("C", "cost"), db.get_data("C", "cap")
    assert isinstance(cost, SetIndexedSeriesData)
    assert isinstance(cap, SetIndexedSeriesData)
    assert cost.values.tolist() == [45, 30]
    assert cap.dims == ("fuel", "seg")
    assert cap.values.tolist() == [[3, 4], [5, 6.5]]


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("{gas: 45, coal: 30}", "{gas: 45}", "do not match"),
        ("{gas: 45, coal: 30}", "{gas: 45, coal: abc}", "must contain numbers"),
        ("{gas: {1: 3, 2: 4}, coal:", "{gas: 3, coal:", "need a mapping"),
        ("cost\n", "cost\n          time-dependent: true\n", "sets only"),
        ("indexed-by: [fuel]\n          value: {", "value: {", "with 'indexed-by'"),
    ],
)
def test_invalid_inline_values(old: str, new: str, message: str) -> None:
    yaml = INLINE_YAML.replace(old, new, 1)
    with pytest.raises(ValueError, match=message):
        build_data_base(parse_yaml_system(io.StringIO(yaml)), None)
