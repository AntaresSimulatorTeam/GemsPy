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
E2E test: bounds of a time sum ``sum(S .. E, X)``, as in Antares Simulator.

A bound is either relative to the current time step (``t``, ``t + ...``,
``t - ...``) or an absolute time index of the block (an expression without
``t``). Both kinds can be mixed, e.g. ``sum(0 .. t, x)`` is a cumulative sum.
"""

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


def test_bound_depending_on_time_raises(tmp_path: Path) -> None:
    study_dir = _make_study(
        tmp_path,
        variables=["x", "a"],
        constraints=["x = d", "a = sum(0 .. d, x)"],
        objective="sum(x)",
        data={"d": [1, 2, 3, 4]},
        constants={},
        nb_time_steps=4,
    )

    with pytest.raises(
        ValueError,
        match="a time sum bound must be fixed in time, got 'd' in constraint 'c1'",
    ):
        load_study(study_dir)


# Objectives computed with antares-modeler 10.1.1 on the same library and system
# (which also gives the same value of gen at every time step).
@pytest.mark.parametrize(
    "constraints, antares_objective",
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
def test_objective_matches_antares(
    tmp_path: Path, constraints: List[str], antares_objective: float
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

    assert _solve(study_dir)["objective-value"] == [antares_objective]
