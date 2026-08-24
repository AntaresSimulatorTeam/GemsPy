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
12-step demand series, supplied in memory rather than through a study folder:
the study directory and its `optim-config.yml` are only an entry point, and
`run_study`'s folder-to-CSV path has its own test (test_study_from_folder.py).

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

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pytest

from gems_craft.optim_config.parsing import (
    ModelOptimConfig,
    OptimConfig,
    OutOfBoundsConstraintConfig,
    OutOfBoundsMode,
    OutOfBoundsProcessingConfig,
    ResolutionConfig,
    ResolutionMode,
    ScenarioScopeConfig,
    TimeScopeConfig,
)
from gems_craft.study.data import TimeSeriesData
from gems_craft.study.folder import load_study
from gems_craft.study.study import Study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.optimization import OptimizationProblem
from gems_runner.simulation.simulation_table import SimulationTable

_STUDY_SRC = Path(__file__).parent / "studies" / "rolling_horizon_suboptimality"

# Aperiodic demand: peaks (4) need gen=2 + discharge=2, mid steps (2) are
# covered by the generator alone, zeros allow recharging.  The pattern is
# deliberately not periodic with the block stride so that the SoC trajectory
# differs between a block's last timestep and the start of the overlap zone.
_DEMAND = [0, 4, 2, 0, 4, 0, 2, 4, 0, 4, 4, 0]
_BLOCK_LENGTH = 6
_BLOCK_OVERLAP = 3


@lru_cache
def _study() -> Study:
    """The committed rolling-horizon study, with its 6-step demand series
    replaced in memory by the 12-step aperiodic one above."""
    study = load_study(_STUDY_SRC)
    study.database.add_data(
        "load_node", "demand", TimeSeriesData(pd.Series(_DEMAND, dtype=float))
    )
    return study


def _config(**resolution: Any) -> OptimConfig:
    """The study's optim-config, with `resolution` built from the given fields.

    Fields left out are left *unset* (not defaulted), which is what
    distinguishes an omitted `carry-over-length` from an explicit `0`, and what
    keeps `block-overlap` — sequential-only — out of the parallel config.
    """
    return OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=0, last_time_step=len(_DEMAND) - 1),
        scenario_scope=ScenarioScopeConfig(include=[0]),
        models=[
            ModelOptimConfig(
                id="rolling-horizon-lib.storage",
                out_of_bounds_processing=OutOfBoundsProcessingConfig(
                    constraints=[
                        OutOfBoundsConstraintConfig(
                            id="soc_balance", mode=OutOfBoundsMode.DROP
                        )
                    ]
                ),
            )
        ],
        resolution=ResolutionConfig(**resolution),
    )


def _solve(config: OptimConfig) -> Tuple[SimulationTable, List[OptimizationProblem]]:
    """Run the study through a `SimulationSession` and return its result table
    plus the solved problems, one per block, in solve order.

    `SimulationSession._run_block` returns the solved problem for carry-over
    extraction *or inspection*, which is what gives the test access to the
    carry-over constraints.
    """
    session = SimulationSession(_study(), config)
    problems: List[OptimizationProblem] = []
    run_block = session._run_block

    def spy(*args: Any, **kwargs: Any) -> Any:
        problem, table = run_block(*args, **kwargs)
        problems.append(problem)
        return problem, table

    session._run_block = spy  # type: ignore[assignment]
    return session.run(), problems


def _pinned_window_lengths(problem: OptimizationProblem) -> Set[int]:
    """Number of timesteps each carry-over equality constraint of `problem`
    fixes.  Empty when the block carries nothing over."""
    linopy_model = problem.linopy_model
    return {
        int(linopy_model.constraints[name].sizes["time"])
        for name in linopy_model.constraints
        if name.startswith("carry_over__")
    }


def _value(
    st: SimulationTable, block: int, component: str, output: str, timestep: int
) -> float:
    df = st.data
    rows = df[
        (df["block"] == block)
        & (df["component"] == component)
        & (df["output"] == output)
        & (df["absolute_time_index"] == timestep)
    ]
    assert len(rows) == 1, (
        f"Expected exactly one row for block={block} component={component} "
        f"output={output} t={timestep}, got {len(rows)}"
    )
    return float(rows.iloc[0]["value"])


@pytest.mark.parametrize(
    "carry_over, expected",
    [
        # Omitted resolves to `block-overlap`: the whole overlap zone is pinned.
        ({}, _BLOCK_OVERLAP),
        ({"carry_over_length": 0}, 0),
        ({"carry_over_length": 1}, 1),
        ({"carry_over_length": 2}, 2),
    ],
)
def test_carry_over_length_fixes_that_many_leading_timesteps(
    carry_over: Dict[str, int], expected: int
) -> None:
    """`carry-over-length: k` fixes, in every block but the first, the k
    leading local timesteps — the k earliest shared timesteps — to the previous
    block's solution, and leaves the rest of the overlap zone free.

    `k = 0` fixes nothing at all: the blocks still overlap (so lag constraints
    keep their history) but are not stitched.
    """
    _, problems = _solve(
        _config(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=_BLOCK_LENGTH,
            block_overlap=_BLOCK_OVERLAP,
            **carry_over,
        )
    )
    # block-length=6, block-overlap=3, t=0..11 → blocks [0..5], [3..8], [6..11]
    # and the truncated tail [9..11].
    assert len(problems) == 4
    assert not _pinned_window_lengths(
        problems[0]
    ), "Nothing is carried into the first block"

    for block_id, problem in enumerate(problems[1:], start=1):
        windows = _pinned_window_lengths(problem)
        assert windows == ({expected} if expected else set()), (
            f"Block {block_id}: every carry-over constraint must fix the "
            f"{expected} leading timesteps, found {sorted(windows)}"
        )


def test_zero_overlap_blocks_fully_independent() -> None:
    """With block-overlap: 0 nothing is carried between blocks: each block is
    solved as if it were alone (no carry-over constraints).

    Two complementary checks:

    - Block 1 ([6..11], demand [2,4,0,4,4,0]) serves its t=7 peak by
      pre-charging its *free* initial storage state.
    - The whole solution is identical to parallel-subproblems mode, which
      solves the same windows independently by construction (and where
      `block-overlap` is not accepted at all).
    """
    seq, _ = _solve(
        _config(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=_BLOCK_LENGTH,
            block_overlap=0,
        )
    )
    par, _ = _solve(
        _config(mode=ResolutionMode.PARALLEL_SUBPROBLEMS, block_length=_BLOCK_LENGTH)
    )

    assert _value(seq, 1, "bus", "unsupplied", 7) == pytest.approx(
        0.0, abs=1e-6
    ), "Block 1 must serve its t=7 peak from a free initial storage state"

    # Both modes enumerate the same windows with the same 0-based block ids.
    for component, output in [
        ("storage", "soc"),
        ("storage", "charge"),
        ("storage", "discharge"),
        ("gen", "p"),
        ("bus", "unsupplied"),
    ]:
        for t in range(len(_DEMAND)):
            block = t // _BLOCK_LENGTH
            v_seq = _value(seq, block, component, output, t)
            v_par = _value(par, block, component, output, t)
            assert v_seq == pytest.approx(v_par, abs=1e-6), (
                f"sequential (block-overlap: 0) and parallel modes disagree at "
                f"t={t} for {component}.{output}: {v_seq} != {v_par}"
            )
