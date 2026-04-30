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
Functional tests for out-of-bounds-processing in optim-config.yml.

Two families of studies are tested.

--- Storage studies (simple_system_*) ---

The ``state_of_charge_balance`` constraint in the storage_unit model
references ``state_of_charge[t-1]``.  On the first timestep (t=0), this
shift falls outside the current 3-step block.

cyclic (default)
    The shifted term wraps around: ``state_of_charge[-1]`` maps to
    ``state_of_charge[2]``.  The storage state is coupled across the
    full block, so the net dispatch is zero and the generator must
    supply the entire load (50 units/step × 3 steps × cost 10 = 1500).

drop
    The constraint is not instantiated at t=0.  The storage can freely
    dispatch up to 40 units at t=0 (no energy balance required there),
    and continues to satisfy the balance at t=1 and t=2.  The generator
    only needs to cover the gap (50 − 40 = 10 units/step × 3 steps
    × cost 10 = 300).

--- Generator commitment studies (system_*_with_param_in_shift) ---

These studies test out-of-bounds processing when the time shift is expressed
via a model parameter (d_min_up, d_min_down) and used within a sum bound rather than a literal integer.
3 time steps, load [10, 10, 100].

gen_1: p_max=90, p_min=10, marginal_cost=100, startup_cost=500,
       d_min_up=2, d_min_down=1.
gen_2: p_max=10, p_min=0,  marginal_cost=10,  startup_cost=0,
       d_min_up=1, d_min_down=1.

cyclic
    All commitment constraints apply at every time step with cyclic wrapping
    (6 instances each for is_on_dynamics, min_up_duration, min_down_duration).
    Optimal: gen_1 off at t=0 (cyclic is_on_dynamics allows this), starts at
    t=1, on at t=2.  gen_2 covers t=0.
    Cost = 10*10 + 100*(10+90) + 10*10 + 500 = 10700.

drop
    Commitment constraints are dropped at out-of-bounds time steps.
    is_on_dynamics (shift -1): dropped at t=0 for both components → 4 instances.
    min_up_duration (sum t-d_min_up+1..t): dropped at t=0 for gen_1 (d_min_up=2)
      but kept for gen_2 (d_min_up=1, no out-of-bounds shift) → 5 instances.
    min_down_duration (sum t-d_min_down+1..t): never dropped (d_min_down=1 for
      both, shift range [0,0] always in-bounds) → 6 instances.
    Gen_1 starts at t=2 only; gen_2 covers t=0 and t=1.
    Cost = 10*10 + 10*10 + 100*90 + 10*10 + 500 = 9800.
"""

import time
from pathlib import Path
from typing import Dict

import linopy
import pytest

from gems.optim_config.parsing import load_optim_config
from gems.simulation import TimeBlock, build_decomposed_problems
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.study.folder import load_study

STUDIES_DIR = Path(__file__).parent / "studies"


def _count_active(model: linopy.Model, name: str) -> int:
    """Return the number of active (non-NaN) rows in a linopy constraint."""
    con = model.constraints[name]
    return int((con.lhs.vars.values >= 0).any(axis=-1).sum())


@pytest.mark.parametrize(
    "study_id, expected_objective",
    [
        ("simple_system_cyclic", 1500.0),
        ("simple_system_drop", 300.0),
        ("system_cyclic_with_param_in_shift", 10700.0),
        ("system_drop_with_param_in_shift", 9800.0),
    ],
)
def test_out_of_bounds_processing(study_id: str, expected_objective: float) -> None:
    study = load_study(STUDIES_DIR / study_id)
    config_path = STUDIES_DIR / study_id / "input" / "optim-config.yml"
    optim_config = load_optim_config(config_path)

    # 3-step block matching the study parameters (first-time-step: 0, last-time-step: 2)
    time_block = TimeBlock(1, [0, 1, 2])
    scenarios = [0]

    decomposed = build_decomposed_problems(study, time_block, scenarios, optim_config)
    decomposed.subproblem.solve(solver_name="highs")

    passed = False
    try:
        assert decomposed.subproblem.termination_condition == "optimal"
        assert decomposed.subproblem.objective_value == pytest.approx(
            expected_objective
        )
        passed = True
    finally:
        if not passed:
            # Dbugging information if the test fails
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            study_path = STUDIES_DIR / study_id
            output_dir = study_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            decomposed.subproblem.export_lp(
                output_dir / f"{study_path.stem}_{timestamp}.lp"
            )
            decomposed.subproblem.export_lp(
                output_dir / f"{study_path.stem}_{timestamp}.mps"
            )
            builder = SimulationTableBuilder(simulation_id=study_path.stem)
            st = builder.build(decomposed.subproblem)
            st.write_csv(
                output_dir=output_dir,
                simulation_id=f"{study_path.stem}_{timestamp}",
                optim_nb=decomposed.subproblem.block.id,
            )


_GEN_PREFIX = "simple_models.generator"

# Expected number of active constraint rows per (study_id, constraint_name).
# Constraints: 2 components (gen_1, gen_2) × 3 timesteps = 6 potential instances each.
#
# system_cyclic_with_param_in_shift — no drop mode, all instances present:
#   is_on_dynamics (lb + ub): 6 each
#   min_up_duration (ub only): 6
#   min_down_duration (ub only): 6
#
# system_drop_with_param_in_shift — drop mode active:
#   is_on_dynamics: shift -1 (from is_on[t-1]) → dropped at t=0 for all → 4
#   min_up_duration: gen_1 has d_min_up=2 → range [-1,0] → dropped at t=0 for gen_1;
#                    gen_2 has d_min_up=1 → range [0,0] → kept at t=0 → 2+3=5
#   min_down_duration: both have d_min_down=1 → range [0,0] → never dropped → 6
_EXPECTED_CONSTRAINT_COUNTS: Dict[str, Dict[str, int]] = {
    "system_cyclic_with_param_in_shift": {
        f"{_GEN_PREFIX}__is_on_dynamics__lb": 6,
        f"{_GEN_PREFIX}__is_on_dynamics__ub": 6,
        f"{_GEN_PREFIX}__min_up_duration__ub": 6,
        f"{_GEN_PREFIX}__min_down_duration__ub": 6,
    },
    "system_drop_with_param_in_shift": {
        f"{_GEN_PREFIX}__is_on_dynamics__lb": 4,
        f"{_GEN_PREFIX}__is_on_dynamics__ub": 4,
        f"{_GEN_PREFIX}__min_up_duration__ub": 5,
        f"{_GEN_PREFIX}__min_down_duration__ub": 6,
    },
}


@pytest.mark.parametrize("study_id", list(_EXPECTED_CONSTRAINT_COUNTS))
def test_constraint_instantiation(study_id: str) -> None:
    """Check that is_on_dynamics, min_up_duration, and min_down_duration are
    instantiated at exactly the expected (component, time) pairs."""
    study = load_study(STUDIES_DIR / study_id)
    config_path = STUDIES_DIR / study_id / "input" / "optim-config.yml"
    optim_config = load_optim_config(config_path)
    time_block = TimeBlock(1, [0, 1, 2])
    decomposed = build_decomposed_problems(study, time_block, [0], optim_config)
    linopy_model = decomposed.subproblem.linopy_model

    expected = _EXPECTED_CONSTRAINT_COUNTS[study_id]
    for constraint_name, expected_count in expected.items():
        actual = _count_active(linopy_model, constraint_name)
        assert actual == expected_count, (
            f"{study_id}: constraint '{constraint_name}' has {actual} active rows, "
            f"expected {expected_count}"
        )
