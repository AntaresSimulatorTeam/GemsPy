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
E2E test: multi-timestep carry-over in sequential mode (Issue #271).

Reuses the rolling_horizon_suboptimality study (generator p_max=2/cost=1,
storage capacity=2/rate=2, bus with ens_cost=100) with a longer, aperiodic
12-step demand series so that the storage state-of-charge trajectory varies
across block boundaries.

Sequential mode with block-length=6, block-overlap=3 over t=0..11:

    block 0: [0  1  2  3  4  5]
    block 1:          [3  4  5  6  7  8]
    block 2:                   [6  7  8  9 10 11]
    block 3:                            [9 10 11]        (truncated tail)

Consecutive blocks share `block-overlap` = 3 absolute timesteps.  The
`carry-over-length` first shared timesteps of block N+1 are pinned to block
N's already-solved values — counted from the *earliest* shared timestep, so
each pinned constraint matches the same absolute timestep in both blocks.
Before the fix for issue #271, the code pinned block N+1's first timestep to
block N's *last* timestep, which is a different absolute timestep whenever
block-overlap >= 2.

The tests assert, from the merged simulation table (which keeps one row per
block for overlapping timesteps), that every pinned shared timestep carries
identical values in both blocks' solutions.
"""

import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from gems_runner.study.runner import run_study

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


def _sequential_config(carry_over_length: str) -> str:
    return _BASE_CONFIG + textwrap.dedent(f"""\
        resolution:
          mode: sequential-subproblems
          block-length: 6
          block-overlap: 3
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


def _assert_pinned_window_consistent(raw: pd.DataFrame, carry_over_length: int) -> None:
    """The first `carry_over_length` shared timesteps of each consecutive block
    pair must have identical values in both blocks, for every output."""
    shared = _shared_timesteps(raw)
    assert shared, "Expected at least two consecutive blocks"
    for (block_n, block_n1), timesteps in shared.items():
        assert timesteps, f"Blocks {block_n} and {block_n1} share no timesteps"
        for t in timesteps[:carry_over_length]:
            for component, output in _OUTPUTS:
                v_prev = _get_value(raw, block_n, component, output, int(t))
                v_next = _get_value(raw, block_n1, component, output, int(t))
                assert v_next == pytest.approx(v_prev, abs=1e-6), (
                    f"Pinned timestep t={t} disagrees between block {block_n} "
                    f"({v_prev}) and block {block_n1} ({v_next}) for "
                    f"{component}.{output}"
                )


def test_full_pin_default(tmp_path: Path) -> None:
    """Omitted carry-over-length defaults to block-overlap: the whole overlap
    zone of every consecutive block pair is pinned to the earlier block's
    values, timestep by absolute timestep."""
    raw = _run(tmp_path, "full_pin", _sequential_config("# carry-over-length omitted"))

    shared = _shared_timesteps(raw)
    # block-length=6, block-overlap=3, t=0..11 → blocks [0..5], [3..8],
    # [6..11], [9..11]; consecutive pairs share exactly 3 timesteps.
    assert shared == {
        (0, 1): [3, 4, 5],
        (1, 2): [6, 7, 8],
        (2, 3): [9, 10, 11],
    }
    _assert_pinned_window_consistent(raw, carry_over_length=3)


def test_explicit_full_pin(tmp_path: Path) -> None:
    """carry-over-length equal to block-overlap behaves like the default."""
    raw = _run(tmp_path, "explicit_full", _sequential_config("carry-over-length: 3"))
    _assert_pinned_window_consistent(raw, carry_over_length=3)


def test_partial_pin(tmp_path: Path) -> None:
    """carry-over-length < block-overlap pins only the leading shared
    timesteps; the rest of the overlap zone is re-optimized freely."""
    raw = _run(tmp_path, "partial_pin", _sequential_config("carry-over-length: 1"))
    _assert_pinned_window_consistent(raw, carry_over_length=1)


def test_zero_carry_over(tmp_path: Path) -> None:
    """Explicit carry-over-length: 0 disables stitching entirely: every block
    is solved independently over its own window, and every timestep of the
    horizon is still present in the output."""
    raw = _run(tmp_path, "zero_carry", _sequential_config("carry-over-length: 0"))

    timesteps = set(raw["absolute_time_index"].dropna().astype(int))
    assert timesteps == set(range(12))


def _no_overlap_config(mode: str) -> str:
    return _BASE_CONFIG + textwrap.dedent(f"""\
        resolution:
          mode: {mode}
          block-length: 6
          block-overlap: 0
    """)


def test_zero_overlap_blocks_fully_independent(tmp_path: Path) -> None:
    """With block-overlap: 0 nothing is carried between blocks: each block is
    solved as if it were alone (no carry-over constraints).

    Two complementary checks:

    - Block 1 ([6..11], demand [2,4,0,4,4,0]) serves its t=7 peak by
      pre-charging its *free* initial storage state.  Under the old implicit
      seeding, SoC(t=6) was pinned to block 0's final SoC=0 and the generator
      (p_max=2, fully used by demand=2 at t=6) could not recharge in time,
      forcing 2 units of unserved energy at t=7.
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
