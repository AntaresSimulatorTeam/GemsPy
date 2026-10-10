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
E2E test: bounds of a time sum ``sum(S .. E, X)``.

A bound is either relative to the current time step (``t``, ``t + ...``,
``t - ...``) or an absolute time index of the block (an expression without
``t``). Both kinds can be mixed, e.g. ``sum(0 .. t, x)`` is a cumulative sum.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession

_PARAMETER = """\
        - id: {id}
          time-dependent: {dependent}
          scenario-dependent: false
"""


def _make_study(
    tmp_path: Path,
    variables: List[str],
    constraints: List[str],
    objective: str,
    data: Dict[str, List[float]],
    constants: Dict[str, float],
    nb_time_steps: int,
    extra_outputs: Optional[Dict[str, str]] = None,
    resolution: str = "",
) -> Path:
    study_dir = tmp_path / "study"
    (study_dir / "input" / "model-libraries").mkdir(parents=True)
    (study_dir / "input" / "data-series").mkdir()

    library = "library:\n  id: lib\n  models:\n    - id: m\n      parameters:\n"
    library += "".join(_PARAMETER.format(id=p, dependent="true") for p in data)
    library += "".join(_PARAMETER.format(id=p, dependent="false") for p in constants)
    library += "      variables:\n" + "".join(
        f"        - id: {v}\n          lower-bound: 0\n" for v in variables
    )
    library += "      constraints:\n" + "".join(
        f"        - id: c{i}\n          expression: {c}\n"
        for i, c in enumerate(constraints)
    )
    if extra_outputs:
        library += "      extra-outputs:\n" + "".join(
            f"        - id: {k}\n          expression: {v}\n"
            for k, v in extra_outputs.items()
        )
    library += (
        "      objective-contributions:\n"
        f"        - id: objective\n          expression: {objective}\n"
    )
    (study_dir / "input" / "model-libraries" / "lib.yml").write_text(library)

    system = "system:\n  id: s\n  components:\n    - id: m1\n      model: lib.m\n"
    system += "      parameters:\n"
    for name in data:
        system += (
            _PARAMETER.format(id=name, dependent="true")
            + f"          value: {name}_ts\n"
        )
        (study_dir / "input" / "data-series" / f"{name}_ts.tsv").write_text(
            "".join(f"{v}\n" for v in data[name])
        )
    for name, value in constants.items():
        system += (
            _PARAMETER.format(id=name, dependent="false")
            + f"          value: {value}\n"
        )
    (study_dir / "input" / "system.yml").write_text(system)

    (study_dir / "input" / "optim-config.yml").write_text(
        f"time-scope:\n  first-time-step: 0\n  last-time-step: {nb_time_steps - 1}\n"
        "solver-options:\n  name: highs\n  logs: false\n" + resolution
    )
    return study_dir


def _solve(study_dir: Path) -> Dict[str, List[float]]:
    optim_config = load_optim_config(study_dir / "input" / "optim-config.yml")
    assert optim_config is not None
    table = SimulationSession(load_study(study_dir), optim_config).run().data
    return {
        output: rows.sort_values("absolute_time_index")["value"].round(6).tolist()
        for output, rows in table.groupby("output")
    }


def _out_of_bounds(mode: str, constraints: List[str]) -> str:
    return (
        "models:\n  - id: lib.m\n    out-of-bounds-processing:\n      constraints:\n"
        + "".join(f"        - id: {c}\n          mode: {mode}\n" for c in constraints)
    )


_SUMS = {
    "a": "sum(0 .. 2, x)",  # absolute bounds
    "c": "sum(0 .. t, x)",  # absolute start, relative end: cumulative sum
    "e": "sum(t .. 3, x)",  # relative start, absolute end
    "r": "sum(t - 1 .. t, x)",  # relative bounds, cyclic in the block
}


def test_absolute_mixed_and_relative_bounds_in_constraints(tmp_path: Path) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", *_SUMS],
        constraints=["x = d", *(f"{v} = {s}" for v, s in _SUMS.items())],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
    )

    results = _solve(study_dir)

    assert results["a"] == [6, 6, 6, 6]
    assert results["c"] == [1, 3, 6, 10]
    assert results["e"] == [10, 9, 7, 4]
    assert results["r"] == [5, 3, 5, 7]


_OUTSIDE_THE_BLOCK = {
    "n": "sum(-1 .. 1, x)",  # negative start: time steps 3, 0, 1
    "p": "sum(2 .. 5, x)",  # end past the block: time steps 2, 3, 0, 1
    "w": "sum(0 .. 5, x)",  # wraps past the end: 0, 1 counted twice
    "m": "sum(-1 .. t, x)",  # mixed: time step 3, then 0 .. t
}


@pytest.mark.parametrize("mode", ["cyclic", "drop"])
def test_absolute_indices_outside_the_block_wrap(tmp_path: Path, mode: str) -> None:
    """An absolute index outside the block wraps around it, as x[N], and never
    causes the constraint to be dropped."""
    study_dir = _make_study(
        tmp_path,
        variables=["x", *_OUTSIDE_THE_BLOCK],
        constraints=["x = d", *(f"{v} = {s}" for v, s in _OUTSIDE_THE_BLOCK.items())],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
        resolution=_out_of_bounds(mode, ["c1", "c2", "c3", "c4"]),
    )

    results = _solve(study_dir)

    assert results["n"] == [7, 7, 7, 7]
    assert results["p"] == [10, 10, 10, 10]
    assert results["w"] == [13, 13, 13, 13]
    assert results["m"] == [5, 7, 10, 14]


def test_absolute_bounds_in_extra_outputs(tmp_path: Path) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x"],
        constraints=["x = d"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
        extra_outputs={"xa": _SUMS["a"], "xc": _SUMS["c"]},
    )

    results = _solve(study_dir)

    assert results["xa"] == [6]  # time-independent
    assert results["xc"] == [1, 3, 6, 10]


def test_absolute_bound_given_by_a_parameter(tmp_path: Path) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", "a = sum(first .. last, x)"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={"first": 1, "last": 2},
        nb_time_steps=4,
    )

    assert _solve(study_dir)["a"] == [5, 5, 5, 5]


@pytest.mark.parametrize(
    "mode, time_sum, expected",
    [
        pytest.param("cyclic", "sum(0 .. d[1], x)", [6, 6, 6, 6], id="absolute"),
        pytest.param("drop", "sum(0 .. d[1], x)", [6, 6, 6, 6], id="absolute-drop"),
        pytest.param("cyclic", "sum(t - (d[0]) .. t, x)", [5, 3, 5, 7], id="relative"),
        # dropped at t = 0, where a is then 0
        pytest.param(
            "drop", "sum(t - (d[0]) .. t, x)", [0, 3, 5, 7], id="relative-drop"
        ),
        pytest.param("cyclic", "sum(t - (d[0]) .. 3, x)", [14, 10, 9, 7], id="mixed"),
        pytest.param("drop", "sum(t - (d[0]) .. 3, x)", [0, 10, 9, 7], id="mixed-drop"),
    ],
)
def test_bound_with_a_time_operator(
    tmp_path: Path, mode: str, time_sum: str, expected: List[float]
) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", f"a = {time_sum}"],
        objective="sum(x) + sum(a)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
        resolution=_out_of_bounds(mode, ["c1"]),
    )

    assert _solve(study_dir)["a"] == expected


@pytest.mark.parametrize(
    "time_sum",
    [
        pytest.param("sum(2 .. 1, x)", id="absolute"),
        pytest.param("sum(t + 2 .. t + 1, x)", id="relative"),
        pytest.param("sum(t + p .. t, x)", id="relative-parameter"),
    ],
)
def test_start_after_end_gives_an_empty_sum(tmp_path: Path, time_sum: str) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", f"a = {time_sum}"],
        objective="sum(x) - sum(a)",
        data={"d": [1, 2, 3, 4]},
        constants={"p": 1},
        nb_time_steps=4,
    )

    assert _solve(study_dir)["a"] == [0, 0, 0, 0]


def test_absolute_index_refers_to_the_block(tmp_path: Path) -> None:
    """In sequential mode, sum(0 .. 1, x) sums the first two steps of each block."""
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", "a = sum(0 .. 1, x)"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
        resolution="resolution:\n  mode: sequential-subproblems\n  block-length: 2\n",
    )

    assert _solve(study_dir)["a"] == [3, 3, 7, 7]


@pytest.mark.parametrize(
    "time_sum, bound",
    [
        pytest.param("sum(0 .. d, x)", "d", id="absolute"),
        pytest.param("sum(t - d .. t, x)", "(t + -(d))", id="relative"),
    ],
)
def test_bound_varying_in_time_raises(
    tmp_path: Path, time_sum: str, bound: str
) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", f"a = {time_sum}"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Model 'lib.m': a time sum bound must be fixed in time, but '{bound}' "
            "varies in time for component(s) m1."
        ),
    ):
        _solve(study_dir)


@pytest.mark.parametrize(
    "mode, expected_r",
    [
        pytest.param("cyclic", [5, 3, 5, 7], id="cyclic"),
        pytest.param("drop", [0, 3, 5, 7], id="drop"),  # dropped at t = 0
    ],
)
def test_bound_declared_time_dependent_but_given_a_constant(
    tmp_path: Path, mode: str, expected_r: List[float]
) -> None:
    """The data decides, in every mode: a parameter declared time-dependent
    can be a bound if the component gives it a constant value."""
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a", "r"],
        constraints=["x = d", "a = sum(0 .. last, x)", "r = sum(t - last .. t, x)"],
        objective="sum(x) + sum(r)",
        data={"d": [1, 2, 3, 4]},
        constants={"last": 1},
        nb_time_steps=4,
        resolution=_out_of_bounds(mode, ["c1", "c2"]),
    )
    library = study_dir / "input" / "model-libraries" / "lib.yml"
    library.write_text(
        library.read_text().replace(
            "        - id: last\n          time-dependent: false",
            "        - id: last\n          time-dependent: true",
        )
    )

    results = _solve(study_dir)

    assert results["a"] == [3, 3, 3, 3]
    assert results["r"] == expected_r


# Expected objective of each study.
@pytest.mark.parametrize(
    "constraints, expected_objective",
    [
        pytest.param(
            [
                "sum(0 .. t, gen) >= sum(0 .. t, demand)",
                "sum(0 .. 2, gen) <= cap3",
                "sum(t-1 .. t, gen) <= ramp2",
            ],
            1331,
            id="mixed-absolute-relative",
        ),
        pytest.param(
            ["gen >= demand", "sum(0 .. 2, gen) >= cap3"], 2873, id="absolute"
        ),
        pytest.param(["sum(0 .. t, gen) >= sum(0 .. t, demand)"], 1256, id="mixed"),
        pytest.param(
            ["sum(t .. 167, gen) >= sum(t .. 167, demand)"], 1275, id="mixed-reversed"
        ),
    ],
)
def test_objective_matches_reference(
    tmp_path: Path, constraints: List[str], expected_objective: float
) -> None:
    nb_time_steps = 168
    study_dir = _make_study(
        tmp_path,
        variables=["gen"],
        constraints=["gen <= pmax", *constraints],
        objective="sum(cost * gen)",
        data={
            "cost": [
                [1, 5, 2, 8, 1, 6, 3][i % 7] + i % 5 for i in range(nb_time_steps)
            ],
            "demand": [[3, 4, 2, 5, 1, 3][i % 6] for i in range(nb_time_steps)],
        },
        constants={"pmax": 10, "cap3": 10, "ramp2": 12},
        nb_time_steps=nb_time_steps,
    )

    assert _solve(study_dir)["objective-value"] == [expected_objective]
