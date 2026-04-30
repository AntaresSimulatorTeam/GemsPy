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
Tests for SimulationTable correctness when the optimization TimeBlock does not
cover the full data horizon.

Test: `test_simtable_on_partial_timeblock`
  - System: 3 components — node, generator (pmax=200), demand (timevarying: demand[t]=t)
  - Data horizon: 150 timesteps
  - TimeBlock: [40, 90)  →  50 timesteps (indices 40–89)
  - Checks that SimulationTable absolute-time-index, block-time-index, and
    generation values are all consistent with the partial block.
"""

import pandas as pd
import pytest

from gems.model.library import Library
from gems.simulation import TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationColumns, SimulationTableBuilder
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    TimeScenarioSeriesData,
    create_component,
)

HORIZON = 150
BLOCK_START = 40
BLOCK_END = 90  # half-open: timesteps 40–89


def test_simtable_on_partial_timeblock(lib_dict_unittest: dict[str, Library]) -> None:
    """
    Verify that SimulationTable reflects absolute time indices [40, 90) when the
    TimeBlock covers only a slice of a 150-timestep data horizon.

    With demand[t] = t and pmax = 200, the optimizer sets generation[t] = t at
    every timestep, making all three assertions self-consistent:
      1. generation values equal their absolute timestep index.
      2. absolute-time-index runs from 40 to 89 (not 0–149 or 0–49).
      3. block-time-index runs from 0 to 49 and equals absolute-time-index − 40.
    """
    node_model = lib_dict_unittest["basic"].models["basic.node"]
    generator_model = lib_dict_unittest["basic"].models["basic.generator"]
    demand_model = lib_dict_unittest["basic"].models["basic.demand"]

    # demand[t] = t for all 150 timesteps
    demand_data = pd.DataFrame(
        list(range(HORIZON)),
        index=range(HORIZON),
        columns=[0],
    )

    database = DataBase()
    database.add_data("G", "p_max", ConstantData(200))
    database.add_data("G", "cost", ConstantData(1))
    database.add_data("D", "demand", TimeScenarioSeriesData(demand_data))

    node = Component(model=node_model, id="N")
    gen = create_component(model=generator_model, id="G")
    demand = create_component(model=demand_model, id="D")

    system = System("test")
    system.add_component(node)
    system.add_component(gen)
    system.add_component(demand)
    system.connect(PortRef(gen, "injection_port"), PortRef(node, "injection_port"))
    system.connect(PortRef(demand, "injection_port"), PortRef(node, "injection_port"))

    time_block = TimeBlock(1, list(range(BLOCK_START, BLOCK_END)))

    problem = build_problem(Study(system, database), time_block, [0])
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"

    # cost=1, generation[t]=t for t in [40, 90) → sum(40..89) = 3225
    assert problem.objective_value == pytest.approx(3225)

    sim_table = SimulationTableBuilder().build(problem)

    # Check 1: fluent API — generation values equal absolute timestep index
    gen_series = sim_table.component("G").output("generation").value(scenario_index=0)
    expected = pd.Series(
        data=[float(t) for t in range(BLOCK_START, BLOCK_END)],
        index=pd.Index(
            range(BLOCK_START, BLOCK_END),
            name=SimulationColumns.ABSOLUTE_TIME_INDEX.value,
        ),
        name=0,
    )
    pd.testing.assert_series_equal(gen_series, expected, check_dtype=False)

    # Check 2: raw DataFrame — both index columns are correct and consistently related
    gen_rows = sim_table.data[
        (sim_table.data[SimulationColumns.COMPONENT.value] == "G")
        & (sim_table.data[SimulationColumns.OUTPUT.value] == "generation")
    ].copy()

    abs_times = sorted(
        gen_rows[SimulationColumns.ABSOLUTE_TIME_INDEX.value].astype(int)
    )
    block_times = sorted(gen_rows[SimulationColumns.BLOCK_TIME_INDEX.value].astype(int))

    assert abs_times == list(range(BLOCK_START, BLOCK_END))
    assert block_times == list(range(BLOCK_END - BLOCK_START))

    # absolute-time-index = block-time-index + BLOCK_START for every row
    offset = (
        gen_rows[SimulationColumns.ABSOLUTE_TIME_INDEX.value].astype(int)
        - gen_rows[SimulationColumns.BLOCK_TIME_INDEX.value].astype(int)
    ).unique()
    assert list(offset) == [BLOCK_START]
