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
E2E test: SimulationTable views on outputs with several blocks, and
objective_values() per resolution mode.

- Sequential mode with overlap (the study of test_sequential_carry_over_length.py:
  block-length 6, overlap 3 over t=0..11) solves time steps 3..8 in two blocks.
- Parallel mode on 13_1 (t=0..3, blocks of 2) solves the time-independent
  investment p_max once per block.

In both cases the views raise unless ``block=`` says which block to read.
"""

from pathlib import Path

import pytest

from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionConfig,
    ResolutionMode,
    ScenarioScopeConfig,
    TimeScopeConfig,
)
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession
from gems_runner.simulation.simulation_table import SimulationTable
from tests.e2e.functional.test_sequential_carry_over_length import (
    _BLOCK_LENGTH,
    _BLOCK_OVERLAP,
    _config,
    _study,
)

_STUDY_13_1 = Path(__file__).parent / "studies" / "13_1"


def _sequential_overlap() -> SimulationTable:
    config = _config(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=_BLOCK_LENGTH,
        block_overlap=_BLOCK_OVERLAP,
    )
    return SimulationSession(_study(), config).run()


def _run_13_1(
    mode: ResolutionMode, scenarios: list, **resolution: int
) -> SimulationTable:
    config = OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=0, last_time_step=3),
        scenario_scope=ScenarioScopeConfig(include=scenarios),
        resolution=ResolutionConfig(mode=mode, **resolution),
    )
    return SimulationSession(load_study(_STUDY_13_1), config).run()


def test_sequential_overlap_needs_block_only_in_the_overlap() -> None:
    st = _sequential_overlap()
    view = st.component("gen").output("p")

    # t=3 is solved in blocks 0 and 1, t=0 only in block 0.
    with pytest.raises(ValueError, match=r"blocks \[0, 1\]"):
        view.value(time_index=3, scenario_index=0)
    view.value(time_index=0, scenario_index=0)

    rows = st.data[(st.data["component"] == "gen") & (st.data["output"] == "p")]
    for block in (0, 1):
        expected = rows[(rows["block"] == block) & (rows["absolute_time_index"] == 3)]
        assert view.value(time_index=3, scenario_index=0, block=block) == (
            pytest.approx(float(expected["value"].iloc[0]))
        )
    assert list(st.component("gen").output("p", block=1).data.index) == list(
        range(3, 9)
    )


def test_parallel_time_independent_output_needs_block() -> None:
    st = _run_13_1(ResolutionMode.PARALLEL_SUBPROBLEMS, [0], block_length=2)
    view = st.component("continuous_generator_candidate").output("p_max")

    with pytest.raises(ValueError, match="block="):
        view.value(scenario_index=0)
    # Each block sizes its own capacity; any time index is accepted.
    assert view.value(time_index=2, scenario_index=0, block=1) == pytest.approx(
        view.value(time_index=0, scenario_index=0, block=1)
    )


def test_objective_values_frontal() -> None:
    objective = _run_13_1(ResolutionMode.FRONTAL, [0, 1]).objective_values()

    assert len(objective) == 1
    assert objective["block"].tolist() == [0]
    assert objective["scenario_index"].isna().all()  # shared by both scenarios


def test_objective_values_parallel() -> None:
    objective = _run_13_1(
        ResolutionMode.PARALLEL_SUBPROBLEMS, [0, 1], block_length=2
    ).objective_values()

    pairs = sorted(zip(objective["scenario_index"], objective["block"]))
    assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_objective_values_sequential() -> None:
    objective = _sequential_overlap().objective_values()

    assert sorted(objective["block"]) == [0, 1, 2, 3]
    assert (objective["scenario_index"] == 0).all()
