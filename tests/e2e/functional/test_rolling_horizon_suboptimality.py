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
E2E test: rolling-horizon suboptimality and sequential carry-over (Issue #102).

System: 1 generator (p_max=2, gen_cost=1) + 1 load (oscillating demand)
        + 1 storage (capacity=2, max_rate=2, η=1, drop-mode SoC balance)
        + 1 bus (power balance with unserved-energy slack, ens_cost=100).

6 time steps (t=0..5):
  demand = [0, 4, 0, 4, 0, 4]   (zero at even steps, peak=4 at odd steps)

The generator alone (p_max=2) cannot cover the full peak demand of 4; the
storage must pre-charge during zero-demand steps so it can contribute 2 units
at each peak step.  Any demand not covered by gen+storage is handled by the
`unsupplied` variable in the bus at a penalty cost of 100/unit.

Sequential mode: block_length=3, block_overlap=1 → blocks [0,1,2], [2,3,4], [4,5].

─── Frontal optimal (total objective = 10) ───────────────────────────────────
With drop mode, SoC[t=0] is a free variable; the optimizer sets it to 2 at no
cost.  Zero-demand even steps are used to recharge the storage; peak odd steps
are served by gen=2 + discharge=2.  No unserved energy.

  t=0: gen=0, discharge=0, charge=0,  SoC=2  (free initial state)
  t=1: gen=2, discharge=2, charge=0,  SoC=0   cost = 2
  t=2: gen=2, discharge=0, charge=2,  SoC=2   cost = 2
  t=3: gen=2, discharge=2, charge=0,  SoC=0   cost = 2
  t=4: gen=2, discharge=0, charge=2,  SoC=2   cost = 2
  t=5: gen=2, discharge=2, charge=0,  SoC=0   cost = 2
  Total generation cost = 10; unserved = 0 → total objective = 10

─── Sequential suboptimality (sum of block objectives = 406) ─────────────────
Block 1 [0,1,2]: SoC[0]=2 free; serves t=1 with gen=2+discharge=2 (cost=2).
  t=2 is the look-ahead step with demand=0.  Within the window no future peak
  is visible, so the optimizer does not recharge → SoC[t=2]=0.
  → carry-over: SoC[t=2] = 0.   Block objective = 2.

Block 2 [2,3,4]: SoC[t=2]=0 fixed by carry-over.
  t=3: peak demand=4 but SoC=0 → gen=2, unsupplied=2.  Cost = 2 + 200 = 202.
  t=4 (look-ahead, demand=0): no recharge incentive → SoC[t=4]=0.
  → carry-over: SoC[t=4] = 0.   Block objective = 202.

Block 3 [4,5]: SoC[t=4]=0 fixed by carry-over.
  t=5: peak demand=4, same situation → gen=2, unsupplied=2. Cost = 2 + 200 = 202.
  Block objective = 202.

  Sum of block objectives = 2 + 202 + 202 = 406.
"""

import shutil
import textwrap
from pathlib import Path

import pytest

from gems.study.runner import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "rolling_horizon_suboptimality"

_BASE_CONFIG = textwrap.dedent("""\
    time-scope:
      first-time-step: 0
      last-time-step: 5
    solver-options:
      name: highs
      logs: false
      parameters: ""
    scenario-scope:
      nb-scenarios: 1
    models:
      - id: rolling-horizon-lib.storage
        out-of-bounds-processing:
          constraints:
            - id: soc_balance
              mode: drop
""")

_FRONTAL_CONFIG = _BASE_CONFIG + textwrap.dedent("""\
    resolution:
      mode: frontal
""")

_SEQUENTIAL_CONFIG = _BASE_CONFIG + textwrap.dedent("""\
    resolution:
      mode: sequential-subproblems
      block-length: 3
      block-overlap: 1
""")


def _run_with_config(study_dir: Path, config_yaml: str):
    import pandas as pd

    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(config_yaml)
    run_study(study_dir)
    output_files = list((study_dir / "output").glob("**/simulation_table_*.csv"))
    assert len(output_files) == 1
    return pd.read_csv(output_files[0])


def _total_objective(raw) -> float:
    """Sum all per-block objective-value entries."""
    return float(raw.loc[raw["output"] == "objective-value", "value"].sum())


def _get_value(raw, component: str, output: str, timestep: int) -> float:
    mask = (
        (raw["component"] == component)
        & (raw["output"] == output)
        & (raw["absolute-time-index"] == timestep)
    )
    rows = raw[mask]
    assert (
        len(rows) >= 1
    ), f"No row for component={component} output={output} t={timestep}"
    return float(rows.iloc[0]["value"])


def test_rolling_horizon_suboptimality(tmp_path: Path) -> None:
    """Sequential mode is suboptimal vs frontal for a storage+oscillating-load system.

    Frontal objective = 10  (storage pre-charged for free via drop mode; covers
    every peak with gen=2 + discharge=2; recharges at cheap zero-demand steps).

    Sequential sum of block objectives = 406  (rolling horizon misses recharge
    opportunities at look-ahead steps → empty storage at peaks → 2 units of
    unserved energy at t=3 and t=5, each penalised at 100/unit).

    The per-timestep assertions pin down exactly where the carry-over state
    diverges between the two modes.
    """
    frontal_dir = tmp_path / "frontal"
    seq_dir = tmp_path / "sequential"
    shutil.copytree(_STUDY_SRC, frontal_dir)
    shutil.copytree(_STUDY_SRC, seq_dir)

    frontal_raw = _run_with_config(frontal_dir, _FRONTAL_CONFIG)
    seq_raw = _run_with_config(seq_dir, _SEQUENTIAL_CONFIG)

    # ── Objective assertions ──────────────────────────────────────────────────
    frontal_obj = _total_objective(frontal_raw)
    seq_obj = _total_objective(seq_raw)

    assert frontal_obj == pytest.approx(
        10.0, rel=1e-6
    ), f"Frontal objective should be 10.0, got {frontal_obj}"
    assert seq_obj == pytest.approx(
        406.0, rel=1e-6
    ), f"Sequential sum of block objectives should be 406.0, got {seq_obj}"
    assert (
        seq_obj > frontal_obj
    ), "Sequential mode must be strictly suboptimal vs frontal"

    # ── SoC carry-over assertions ─────────────────────────────────────────────
    # Frontal recharges at t=2 and t=4 (zero-demand steps); sequential does not
    # because the look-ahead window never reveals the upcoming peak.
    assert _get_value(frontal_raw, "storage", "soc", 2) == pytest.approx(
        2.0, rel=1e-6
    ), "Frontal: storage should be fully recharged (SoC=2) at t=2"
    assert _get_value(seq_raw, "storage", "soc", 2) == pytest.approx(
        0.0, abs=1e-6
    ), "Sequential: storage should be empty (SoC=0) at t=2 — missed recharge"

    assert _get_value(frontal_raw, "storage", "soc", 4) == pytest.approx(
        2.0, rel=1e-6
    ), "Frontal: storage should be fully recharged (SoC=2) at t=4"
    assert _get_value(seq_raw, "storage", "soc", 4) == pytest.approx(
        0.0, abs=1e-6
    ), "Sequential: storage should be empty (SoC=0) at t=4 — missed recharge"

    # ── Unserved energy assertions ────────────────────────────────────────────
    # Frontal has zero unserved energy at every step (storage covers the gap).
    # Sequential cannot discharge at t=3 and t=5 (carry-over SoC=0) → 2 units
    # of unserved energy at each of those steps.
    assert _get_value(frontal_raw, "bus", "unsupplied", 3) == pytest.approx(
        0.0, abs=1e-6
    ), "Frontal: no unserved energy at t=3 (storage discharges)"
    assert _get_value(seq_raw, "bus", "unsupplied", 3) == pytest.approx(
        2.0, rel=1e-6
    ), "Sequential: 2 units unserved at t=3 (storage empty after missed recharge)"

    assert _get_value(frontal_raw, "bus", "unsupplied", 5) == pytest.approx(
        0.0, abs=1e-6
    ), "Frontal: no unserved energy at t=5 (storage discharges)"
    assert _get_value(seq_raw, "bus", "unsupplied", 5) == pytest.approx(
        2.0, rel=1e-6
    ), "Sequential: 2 units unserved at t=5 (storage empty after missed recharge)"
