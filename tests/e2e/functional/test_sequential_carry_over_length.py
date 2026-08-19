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
E2E test: multi-timestep carry-over in sequential mode.

Reuses the rolling_horizon_suboptimality study (generator p_max=2/cost=1,
storage capacity=2/rate=2, bus with ens_cost=100) with a longer, aperiodic
12-step demand series.

Sequential mode with block-length=6, block-overlap=3 over t=0..11:

    block 0: [0  1  2  3  4  5]
    block 1:          [3  4  5  6  7  8]
    block 2:                   [6  7  8  9 10 11]
    block 3:                            [9 10 11]        (truncated tail)

Consecutive blocks share `block-overlap` = 3 absolute timesteps.  The
`carry-over-length` first shared timesteps of block N+1 are pinned to block
N's already-solved values — counted from the *earliest* shared timestep, so
each pinned constraint matches the same absolute timestep in both blocks.

`carry-over-length` is checked on the carry-over constraints themselves: what
the setting controls is which variables are *fixed* and which are left free,
and comparing solved values cannot tell a free timestep that happens to
re-optimize to the same value from a fixed one.
"""

import shutil
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Set

import pandas as pd
import pytest
import xarray as xr

from gems_craft.expression.expression import literal, param, var
from gems_craft.model import Constraint, float_parameter, float_variable, model
from gems_craft.optim_config.parsing import load_optim_config
from gems_craft.study import ConstantData, DataBase, Study, System, create_component
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.optimization import (
    OptimizationProblem,
    _validate_initial_values,
)
from gems_runner.study.runner import run_study
from tests.e2e.functional.libs.standard import CONSTANT

_STUDY_SRC = Path(__file__).parent / "studies" / "rolling_horizon_suboptimality"

# Aperiodic demand: peaks (4) need gen=2 + discharge=2, mid steps (2) are
# covered by the generator alone, zeros allow recharging.  The pattern is
# deliberately not periodic with the block stride so that the SoC trajectory
# differs between a block's last timestep and the start of the overlap zone.
_DEMAND_12 = [0, 4, 2, 0, 4, 0, 2, 4, 0, 4, 4, 0]

_BASE_CONFIG = textwrap.dedent("""\
    time-scope:
      first-time-step: 0
      last-time-step: 11
    solver-options:
      name: highs
      logs: false
      parameters: ""
    scenario-scope:
      include:
        - 0
    models:
      - id: rolling-horizon-lib.storage
        out-of-bounds-processing:
          constraints:
            - id: soc_balance
              mode: drop
""")


_BLOCK_LENGTH = 6
_BLOCK_OVERLAP = 3


def _sequential_config(carry_over_length: str) -> str:
    return _BASE_CONFIG + textwrap.dedent(f"""\
        resolution:
          mode: sequential-subproblems
          block-length: {_BLOCK_LENGTH}
          block-overlap: {_BLOCK_OVERLAP}
          {carry_over_length}
    """)


_OUTPUTS = [
    ("storage", "soc"),
    ("storage", "charge"),
    ("storage", "discharge"),
    ("gen", "p"),
    ("bus", "unsupplied"),
]


def _run(tmp_path: Path, name: str, config_yaml: str) -> pd.DataFrame:
    study_dir = tmp_path / name
    shutil.copytree(_STUDY_SRC, study_dir)
    demand_path = study_dir / "input" / "data-series" / "demand.txt"
    demand_path.write_text("\n".join(str(d) for d in _DEMAND_12) + "\n")
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(config_yaml)
    run_study(study_dir)
    output_files = list((study_dir / "output").glob("**/simulation_table_*.csv"))
    assert len(output_files) == 1
    return pd.read_csv(output_files[0])


def _get_value(
    raw: pd.DataFrame, block: int, component: str, output: str, timestep: int
) -> float:
    mask = (
        (raw["block"] == block)
        & (raw["component"] == component)
        & (raw["output"] == output)
        & (raw["absolute_time_index"] == timestep)
    )
    rows = raw[mask]
    assert len(rows) == 1, (
        f"Expected exactly one row for block={block} component={component} "
        f"output={output} t={timestep}, got {len(rows)}"
    )
    return float(rows.iloc[0]["value"])


def _shared_timesteps(raw: pd.DataFrame) -> dict:
    """Map each consecutive block pair (n, n+1) to their shared absolute timesteps."""
    times_by_block = {
        int(b): set(raw.loc[raw["block"] == b, "absolute_time_index"].dropna())
        for b in raw["block"].unique()
    }
    blocks = sorted(times_by_block)
    return {
        (n, n + 1): sorted(times_by_block[n] & times_by_block[n + 1])
        for n in blocks[:-1]
    }


def _run_sequential_session(
    tmp_path: Path, name: str, config_yaml: str
) -> List[OptimizationProblem]:
    """Run the study through a `SimulationSession` and return the solved
    problems, one per block, in solve order.

    `run_study` drops them; the session hands each one back
    (`SimulationSession._run_block` returns the solved problem for carry-over
    extraction *or inspection*), which is what gives the test access to the
    carry-over constraints.
    """
    study_dir = tmp_path / name
    shutil.copytree(_STUDY_SRC, study_dir)
    demand_path = study_dir / "input" / "data-series" / "demand.txt"
    demand_path.write_text("\n".join(str(d) for d in _DEMAND_12) + "\n")
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(config_yaml)

    optim_config = load_optim_config(config_path)
    assert optim_config is not None
    session = SimulationSession(load_study(study_dir), optim_config)

    problems: List[OptimizationProblem] = []
    run_block = session._run_block

    def spy(block: TimeBlock, **kwargs: Any) -> Any:
        problem, table = run_block(block, **kwargs)
        problems.append(problem)
        return problem, table

    session._run_block = spy  # type: ignore[assignment]
    session.run()
    return problems


def _pinned_window_lengths(problem: OptimizationProblem) -> Set[int]:
    """Number of timesteps each carry-over equality constraint of `problem`
    fixes.  Empty when the block carries nothing over."""
    linopy_model = problem.linopy_model
    return {
        int(linopy_model.constraints[name].sizes["time"])
        for name in linopy_model.constraints
        if name.startswith("carry_over__")
    }


@pytest.fixture(params=[None, 0, 1, 2], ids=["omitted", "0", "1", "2"])
def carry_over_length(request: pytest.FixtureRequest) -> Optional[int]:
    """`carry-over-length` under test: explicitly 0, 1 or 2, or omitted from
    the config — which resolves to `block-overlap`."""
    return request.param


def test_carry_over_length_fixes_that_many_leading_timesteps(
    tmp_path: Path, carry_over_length: Optional[int]
) -> None:
    """`carry-over-length: k` fixes, in every block but the first, the k
    leading local timesteps — the k earliest shared timesteps — to the previous
    block's solution, and leaves the rest of the overlap zone free.

    `k = 0` fixes nothing at all: the blocks still overlap (so lag constraints
    keep their history) but are not stitched.
    """
    setting = (
        "# carry-over-length omitted"
        if carry_over_length is None
        else f"carry-over-length: {carry_over_length}"
    )
    expected = _BLOCK_OVERLAP if carry_over_length is None else carry_over_length

    problems = _run_sequential_session(
        tmp_path, f"carry_over_{carry_over_length}", _sequential_config(setting)
    )
    # block-length=6, block-overlap=3, t=0..11 → blocks [0..5], [3..8], [6..11]
    # and the truncated tail [9..11].
    assert len(problems) == 4

    assert not _pinned_window_lengths(
        problems[0]
    ), "Nothing is carried into the first block"

    for block_id, problem in enumerate(problems[1:], start=1):
        windows = _pinned_window_lengths(problem)
        if expected == 0:
            assert not windows, (
                f"Block {block_id}: 'carry-over-length: 0' must leave the whole "
                f"overlap zone free, found constraints fixing {sorted(windows)} "
                f"timestep(s)"
            )
        else:
            assert windows == {expected}, (
                f"Block {block_id}: every carry-over constraint must fix the "
                f"{expected} leading timesteps, found {sorted(windows)}"
            )


def _no_overlap_config(mode: str) -> str:
    # `block-overlap` is sequential-only and rejected in other modes; parallel
    # partitions the horizon by construction, which is the same window layout.
    overlap = "  block-overlap: 0\n" if mode == "sequential-subproblems" else ""
    return _BASE_CONFIG + textwrap.dedent(f"""\
        resolution:
          mode: {mode}
          block-length: 6
    """) + overlap


def test_zero_overlap_blocks_fully_independent(tmp_path: Path) -> None:
    """With block-overlap: 0 nothing is carried between blocks: each block is
    solved as if it were alone (no carry-over constraints).

    Two complementary checks:

    - Block 1 ([6..11], demand [2,4,0,4,4,0]) serves its t=7 peak by
      pre-charging its *free* initial storage state.
    - The whole solution is identical to parallel-subproblems mode, which
      solves the same windows independently by construction.
    """
    seq_raw = _run(
        tmp_path, "seq_no_overlap", _no_overlap_config("sequential-subproblems")
    )
    par_raw = _run(tmp_path, "parallel", _no_overlap_config("parallel-subproblems"))

    # block-length=6, block-overlap=0, t=0..11 → blocks [0..5] and [6..11]
    # share no timesteps.
    assert _shared_timesteps(seq_raw) == {(0, 1): []}

    # No carry-over constraint: block 1's initial SoC is free, the t=7 peak
    # is fully served.
    assert _get_value(seq_raw, 1, "bus", "unsupplied", 7) == pytest.approx(
        0.0, abs=1e-6
    ), "Block 1 must serve its t=7 peak from a free initial storage state"

    # Fully independent blocks: identical to parallel-subproblems mode
    # (both modes enumerate the same windows with the same 0-based block ids).
    for component, output in _OUTPUTS:
        for t in range(12):
            v_seq = _get_value(seq_raw, t // 6, component, output, t)
            v_par = _get_value(par_raw, t // 6, component, output, t)
            assert v_seq == pytest.approx(v_par, abs=1e-6), (
                f"sequential (block-overlap: 0) and parallel modes disagree at "
                f"t={t} for {component}.{output}: {v_seq} != {v_par}"
            )


# ---------------------------------------------------------------------------
# `initial_values` contract, at the `build_problem` level
#
# The tests above are run-level: they observe the pinned values, not the
# constraints that produced them.  These two build a problem directly to check
# *which* variables the carry-over pins, and that a malformed `initial_values`
# is rejected before anything is built.
# ---------------------------------------------------------------------------


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
