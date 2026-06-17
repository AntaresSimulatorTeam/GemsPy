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
from pathlib import Path

import pytest
from libs.standard import BALANCE_PORT_TYPE, DEMAND_MODEL, GENERATOR_MODEL

from gems.model.library import Library, library
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    create_component,
)


@pytest.fixture
def std_lib() -> Library:
    return library(
        id="std", port_types=[BALANCE_PORT_TYPE], models=[GENERATOR_MODEL, DEMAND_MODEL]
    )


@pytest.fixture
def ac_lib(libs_dir: Path, std_lib: Library) -> dict[str, Library]:
    lib_file = libs_dir / "ac.yml"
    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)
        return resolve_library([input_lib], preloaded_libs=[std_lib])


def test_ac_network_no_links(ac_lib: dict[str, Library]) -> None:
    """
    The system only has one AC node where a generator and a demand are connected.

    There is actually no AC link connected to it, we just check that
    generation matches demand on this node:
     - demand = 100
     - cost = 30
     --> objective = 30 * 100 = 3000
    """
    ac_node_model = ac_lib["ac"].models["ac.ac-node"]

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(30))

    node = Component(model=ac_node_model, id="N")
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
    system.connect(PortRef(demand, "balance_port"), PortRef(node, "injections"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node, "injections"))

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(3000, abs=0.01)


def test_ac_network(ac_lib: dict[str, Library]) -> None:
    """
    The system only has 2 AC nodes connected by 1 AC link.

    Node 1 carries the demand of 100 MW,
    node 2 carries the generator with a cost of 35 per MWh.

    We check that final cost matches the demand: 100 * 35 = 3500,
    and that flow on the line is -100 MW.
    """
    ac_node_model = ac_lib["ac"].models["ac.ac-node"]
    ac_link_model = ac_lib["ac"].models["ac.ac-link"]

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(35))

    database.add_data("L", "reactance", ConstantData(1))

    node1 = Component(model=ac_node_model, id="1")
    node2 = Component(model=ac_node_model, id="2")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )

    gen = create_component(
        model=GENERATOR_MODEL,
        id="G",
    )

    link = create_component(
        model=ac_link_model,
        id="L",
    )

    system = System("test")
    system.add_component(node1)
    system.add_component(node2)
    system.add_component(demand)
    system.add_component(gen)
    system.add_component(link)
    system.connect(PortRef(demand, "balance_port"), PortRef(node1, "injections"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node2, "injections"))
    system.connect(PortRef(link, "port1"), PortRef(node1, "links"))
    system.connect(PortRef(link, "port2"), PortRef(node2, "links"))

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(3500, abs=0.01)

    df = SimulationTableBuilder().build(problem)
    assert df.component("L").output("flow").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-100, abs=0.01)


def test_parallel_ac_links(ac_lib: dict[str, Library]) -> None:
    """
    The system has 2 AC nodes connected by 2 parallel links,
    where reactance is 1 for line L1, and 2 for line L2.
    We expect flow to be te twice bigger on L1 than on L2.

    Node 1 carries the demand of 100 MW,
    node 2 carries the generator with a cost of 35 per MWh.

    We check that final cost matches the demand: 100 * 35 = 3500,
    and that flow on L1 is -66. MW while flow on L2 is only -33.3 MW.
    """
    ac_node_model = ac_lib["ac"].models["ac.ac-node"]
    ac_link_model = ac_lib["ac"].models["ac.ac-link"]

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(35))

    database.add_data("L1", "reactance", ConstantData(1))
    database.add_data("L2", "reactance", ConstantData(2))

    node1 = Component(model=ac_node_model, id="1")
    node2 = Component(model=ac_node_model, id="2")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )
    gen = create_component(
        model=GENERATOR_MODEL,
        id="G",
    )
    link1 = create_component(
        model=ac_link_model,
        id="L1",
    )
    link2 = create_component(
        model=ac_link_model,
        id="L2",
    )

    system = System("test")
    system.add_component(node1)
    system.add_component(node2)
    system.add_component(demand)
    system.add_component(gen)
    system.add_component(link1)
    system.add_component(link2)
    system.connect(PortRef(demand, "balance_port"), PortRef(node1, "injections"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node2, "injections"))
    system.connect(PortRef(link1, "port1"), PortRef(node1, "links"))
    system.connect(PortRef(link1, "port2"), PortRef(node2, "links"))
    system.connect(PortRef(link2, "port1"), PortRef(node1, "links"))
    system.connect(PortRef(link2, "port2"), PortRef(node2, "links"))

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(3500, abs=0.01)

    df = SimulationTableBuilder().build(problem)
    assert df.component("L1").output("flow").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-66.67, abs=0.01)
    assert df.component("L2").output("flow").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-33.33, abs=0.01)


def test_parallel_ac_links_with_pst(ac_lib: dict[str, Library]) -> None:
    """
    Same case as in parallel_ac_links but:
     - flow is restricted to 50 MW on line L1, so it cannot
       anymore transfer 66,7MW
     - flow can be influenced by a phase shifter on line L2

    We expect the case to be feasible thanks to the phase shifter,
    which will allow to balance the flow between the 2 lines.
    Therefore we expect flows to be 50 MW on both lines.

    Objective value is 3500 (for generation) + 50 (for phase shift).
    """
    ac_node_model = ac_lib["ac"].models["ac.ac-node"]
    ac_link_model = ac_lib["ac"].models["ac.ac-link-with-limit"]
    pst_model = ac_lib["ac"].models["ac.ac-link-with-pst"]

    database = DataBase()
    database.add_data("D", "demand", ConstantData(100))

    database.add_data("G", "p_max", ConstantData(100))
    database.add_data("G", "cost", ConstantData(35))

    database.add_data("L", "reactance", ConstantData(1))
    database.add_data("L", "flow_limit", ConstantData(50))
    database.add_data("T", "reactance", ConstantData(2))
    database.add_data("T", "flow_limit", ConstantData(50))
    database.add_data("T", "phase_shift_cost", ConstantData(1))

    node1 = Component(model=ac_node_model, id="1")
    node2 = Component(model=ac_node_model, id="2")
    demand = create_component(
        model=DEMAND_MODEL,
        id="D",
    )
    gen = create_component(
        model=GENERATOR_MODEL,
        id="G",
    )
    link1 = create_component(
        model=ac_link_model,
        id="L",
    )
    link2 = create_component(
        model=pst_model,
        id="T",
    )

    system = System("test")
    system.add_component(node1)
    system.add_component(node2)
    system.add_component(demand)
    system.add_component(gen)
    system.add_component(link1)
    system.add_component(link2)
    system.connect(PortRef(demand, "balance_port"), PortRef(node1, "injections"))
    system.connect(PortRef(gen, "balance_port"), PortRef(node2, "injections"))
    system.connect(PortRef(link1, "port1"), PortRef(node1, "links"))
    system.connect(PortRef(link1, "port2"), PortRef(node2, "links"))
    system.connect(PortRef(link2, "port1"), PortRef(node1, "links"))
    system.connect(PortRef(link2, "port2"), PortRef(node2, "links"))

    scenarios = 1
    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), list(range(scenarios))
    )
    problem.solve(solver_name="highs")
    assert problem.termination_condition == "optimal"
    assert problem.objective_value == pytest.approx(3550, abs=0.01)

    df = SimulationTableBuilder().build(problem)
    assert df.component("L").output("flow").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-50, abs=0.01)
    assert df.component("T").output("flow").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-50, abs=0.01)
    assert df.component("T").output("phase_shift").value(
        time_index=0, scenario_index=0
    ) == pytest.approx(-50, abs=0.01)
