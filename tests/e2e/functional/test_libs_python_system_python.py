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
This module contains end-to-end functional tests for systems built by:
- Using Python models,
- Building the system object in Python.

Several cases are tested:

1. **Basic Balance**:
    - Description: Balance on a single node with fixed demand and generation.
    - Name: `test_basic_balance`.

2. **Time Series Data**:
    - Description: Multiple timesteps with varying demand.
    - Name: `test_timeseries`.

3. **Variable Bounds**:
    - Description: Generator models with variable bounds, including feasibility and infeasibility cases.
    - Name: `test_variable_bound`.

4. **Short-Term Storage**:
    - Description: Short-term storage behavior over different horizons and efficiencies.
    - Names: `test_short_test_horizon_10`, `test_short_test_horizon_5`.
"""

import pandas as pd
import pytest

from gems.expression import literal, param, var
from gems.expression.indexing_structure import IndexingStructure
from gems.model import Model, ModelPort, float_parameter, float_variable, model
from gems.model.port import PortFieldDefinition, PortFieldId
from gems.simulation import TimeBlock, build_problem
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    TimeScenarioIndex,
    TimeScenarioSeriesData,
    create_component,
)
from tests.e2e.functional.libs.standard import (
    BALANCE_PORT_TYPE,
    DEMAND_MODEL,
    GENERATOR_MODEL,
    NODE_BALANCE_MODEL,
    SHORT_TERM_STORAGE_SIMPLE,
    SPILLAGE_MODEL,
    UNSUPPLIED_ENERGY_MODEL,
)


def test_basic_balance() -> None:
    """
    Balance on one node with one fixed demand and one generation, on 1 timestep.
    """

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))

    node = Component(model=NODE_BALANCE_MODEL, id="N")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )

    gen = create_component(
        model=GENERATOR_MODEL,
        id="G",
    )

    system = System("test")
    system.add_component(node)
    system.add_component(demand)
    system.add_component(gen)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 3000


def test_timeseries() -> None:
    """
    Basic case with 2 timesteps, where the demand is 100 on first timestep and 50 on second timestep.
    """

    database = DataBase()

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))
    demand_data = pd.DataFrame(
        [
            [100],
            [50],
        ],
        index=[0, 1],
        columns=[0],
    )

    demand_time_scenario_series = TimeScenarioSeriesData(demand_data)
    database.add_data("D", "demand", demand_time_scenario_series)

    node = Component(model=NODE_BALANCE_MODEL, id="1")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )

    gen = create_component(
        model=GENERATOR_MODEL,
        id="G",
    )

    system = System("test")
    system.add_component(node)
    system.add_component(demand)
    system.add_component(gen)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))

    time_block = TimeBlock(1, [0, 1])
    scenarios = 1

    problem = build_problem(Study(system, database), time_block, list(range(scenarios)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 100 * 30 + 50 * 30


def create_one_node_network(generator_model: Model) -> System:
    node = Component(model=NODE_BALANCE_MODEL, id="1")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )

    gen = create_component(
        model=generator_model,
        id="G",
    )

    system = System("test")
    system.add_component(node)
    system.add_component(demand)
    system.add_component(gen)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node, "balance_port"))
    return system


def create_simple_database(max_generation: float = 100) -> DataBase:
    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(max_generation))
    database.add_data("G", "cost", ConstantData(30))
    return database


def test_variable_bound() -> None:
    """
    Create a system with one node, one demand and one generator on this node.
    Demand is constant 100, cost of generation is constant 30.
    Max generation can be chosen to make it infeasible or not.
    Variation of generator model using variable bound instead of constraint.
    """

    generator_model = model(
        id="GEN_WITH_VARIABLE_BOUND",
        parameters=[
            float_parameter("p_max", IndexingStructure(False, False)),
            float_parameter("cost", IndexingStructure(False, False)),
        ],
        variables=[
            float_variable(
                "generation",
                lower_bound=literal(0),
                upper_bound=param("p_max"),
            )
        ],
        ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId("balance_port", "flow"),
                definition=var("generation"),
            )
        ],
        objective_contributions={
            "operational": (param("cost") * var("generation")).time_sum().expec()
        },
    )

    system = create_one_node_network(generator_model)
    database = create_simple_database(max_generation=200)
    problem = build_problem(Study(system, database), TimeBlock(1, [0]), list(range(1)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == 3000

    system = create_one_node_network(generator_model)
    database = create_simple_database(max_generation=80)
    problem = build_problem(Study(system, database), TimeBlock(1, [0]), list(range(1)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "infeasible"  # Infeasible

    system = create_one_node_network(generator_model)
    database = create_simple_database(max_generation=0)  # Equal upper and lower bounds
    problem = build_problem(Study(system, database), TimeBlock(1, [0]), list(range(1)))
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "infeasible"

    system = create_one_node_network(generator_model)
    database = create_simple_database(max_generation=-10)
    with pytest.raises(
        ValueError,
        match=r"Upper bound \(-10\) must be strictly greater than lower bound \(0\) for variable G.generation",
    ):
        problem = build_problem(
            Study(system, database), TimeBlock(1, [0]), list(range(1))
        )


def generate_data(
    efficiency: float, horizon: int, scenarios: int
) -> TimeScenarioSeriesData:
    data = {}
    for scenario in range(scenarios):
        for absolute_timestep in range(horizon):
            if absolute_timestep == 0:
                data[TimeScenarioIndex(absolute_timestep, scenario)] = -18.0
            else:
                data[TimeScenarioIndex(absolute_timestep, scenario)] = 2 * efficiency

    values = [value for value in data.values()]
    data_df = pd.DataFrame(values, columns=["Value"])
    return TimeScenarioSeriesData(data_df)


def short_term_storage_base(efficiency: float, horizon: int) -> None:
    # 18 produced in the 1st time-step, then consumed 2 * efficiency in the rest
    time_blocks = [TimeBlock(0, list(range(horizon)))]
    scenarios = 1
    database = DataBase()

    database.add_data("D", "demand", generate_data(efficiency, horizon, scenarios))

    database.add_data("U", "cost", ConstantData(10))
    database.add_data("S", "cost", ConstantData(1))

    database.add_data("STS1", "p_max_injection", ConstantData(100))
    database.add_data("STS1", "p_max_withdrawal", ConstantData(50))
    database.add_data("STS1", "level_min", ConstantData(0))
    database.add_data("STS1", "level_max", ConstantData(1000))
    database.add_data("STS1", "inflows", ConstantData(0))
    database.add_data("STS1", "efficiency", ConstantData(efficiency))

    node = Component(model=NODE_BALANCE_MODEL, id="1")
    spillage = create_component(model=SPILLAGE_MODEL, id="S")

    unsupplied = create_component(model=UNSUPPLIED_ENERGY_MODEL, id="U")

    demand = create_component(model=DEMAND_MODEL, id="D")

    short_term_storage = create_component(
        model=SHORT_TERM_STORAGE_SIMPLE,
        id="STS1",
    )

    system = System("test")
    system.add_component(node)
    for component in [demand, short_term_storage, spillage, unsupplied]:
        system.add_component(component)
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    system.connect(
        PortRef(short_term_storage, "balance_port"), PortRef(node, "balance_port")
    )
    system.connect(PortRef(spillage, "balance_port"), PortRef(node, "balance_port"))
    system.connect(PortRef(unsupplied, "balance_port"), PortRef(node, "balance_port"))

    problem = build_problem(
        Study(system, database),
        time_blocks[0],
        list(range(scenarios)),
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"

    # The short-term storage should satisfy the load
    # No spillage / unsupplied energy is expected
    assert problem.objective_value == pytest.approx(0, abs=0.01)

    # TODO: update variable access


def test_short_test_horizon_10() -> None:
    short_term_storage_base(efficiency=0.8, horizon=10)


def test_short_test_horizon_5() -> None:
    short_term_storage_base(efficiency=0.2, horizon=5)
