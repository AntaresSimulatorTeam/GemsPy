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

import pytest

from gems.expression import literal, param, var
from gems.expression.indexing_structure import IndexingStructure
from gems.model.common import ProblemContext
from gems.model.constraint import Constraint
from gems.model.model import ModelPort, model
from gems.model.parameter import float_parameter
from gems.model.port import PortField, PortFieldDefinition, PortFieldId, PortType
from gems.model.variable import float_variable
from gems.simulation import BendersSolution, TimeBlock, build_benders_decomposed_problem
from gems.simulation.decision_tree import (
    DecisionTreeNode,
    InterDecisionTimeScenarioConfig,
)
from gems.study.data import ConstantData, DataBase, TreeData
from gems.study.network import Component, Network, Node, PortRef, create_component
from tests.e2e.functional.libs.standard import (
    DEMAND_MODEL,
    GENERATOR_MODEL,
    NODE_WITH_SPILL_AND_ENS,
)


@pytest.fixture
def generator() -> Component:
    generator = create_component(
        model=GENERATOR_MODEL,
        id="BASE",
    )
    return generator


CONSTANT = IndexingStructure(False, False)
BALANCE_PORT_TYPE = PortType(id="balance", fields=[PortField("flow")])


@pytest.fixture
def candidate() -> Component:
    candidate = create_component(
        model=model(
            id="GEN_WITH_INSTALLED_CAPA",
            parameters=[
                float_parameter("op_cost", CONSTANT),
                float_parameter("invest_cost", CONSTANT),
                float_parameter("max_invest", CONSTANT),
                float_parameter("installed_capa", CONSTANT),
            ],
            variables=[
                float_variable("generation", lower_bound=literal(0)),
                float_variable(
                    "invested_capa",
                    lower_bound=literal(0),
                    structure=CONSTANT,
                    context=ProblemContext.COUPLING,
                ),
                float_variable(
                    "delta_invest",
                    lower_bound=literal(0),
                    upper_bound=param("max_invest"),
                    structure=CONSTANT,
                    context=ProblemContext.INVESTMENT,
                ),
            ],
            ports=[
                ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port"),
            ],
            port_fields_definitions=[
                PortFieldDefinition(
                    port_field=PortFieldId("balance_port", "flow"),
                    definition=var("generation"),
                ),
            ],
            constraints=[
                Constraint(
                    name="Max generation",
                    expression=var("generation")
                    <= param("installed_capa") + var("invested_capa"),
                )
            ],
            objective_operational_contribution=(param("op_cost") * var("generation"))
            .time_sum()
            .expec(),
            objective_investment_contribution=param("invest_cost")
            * var("delta_invest"),
        ),
        id="CAND",
    )
    return candidate


@pytest.fixture
def demand() -> Component:
    demand = create_component(model=DEMAND_MODEL, id="D")
    return demand


@pytest.fixture
def node() -> Node:
    node = Node(model=NODE_WITH_SPILL_AND_ENS, id="N")
    return node


def test_investment_pathway_on_sequential_nodes(
    node: Node,
    demand: Component,
    candidate: Component,
) -> None:
    """
    A first simple test on the investment pathway
    Here, only two nodes are represented, a parent and a child nodes
    with probability one of going from parent to child

    The goal here is to show that, in the parent node, the demand is already met
    by the existing fixed production. However, for the child node without any new
    investment, it would create some unsupplied energy, which is very expensive.

    The investment on the child node, even though enough for the demand, is also
    more expensive than on the parent (this should represent a late investment fee).

    To minimize the expected cost in this 2-node tree, one should expect the maximum
    investment on the parent node, and the rest on the child node.

    Here below the values used:

                         PARENT    |    CHILD
         Demand (MW):     100            200
     Fixed prod (MW):     100            100
     Max invest (MW):      80            100
        Op cost  ($):      10             10
    Invest cost  ($):     100            300
       ENS cost  ($):   10000          10000

    The solution should be:

                  prob   |   investment  |  operational
    parent:          1 x [     100 x 80  +     10 x 100 ]
    child :        + 1 x [     300 x 20  +     10 x 200 ]
                   = 17 000

    """
    # === Populating Database ===
    database = DataBase()
    database.add_data("N", "spillage_cost", ConstantData(10))
    database.add_data("N", "ens_cost", ConstantData(10000))

    database.add_data(
        "D",
        "demand",
        TreeData(
            {
                "parent": ConstantData(100),
                "child": ConstantData(200),
            }
        ),
    )

    database.add_data("CAND", "op_cost", ConstantData(10))
    database.add_data("CAND", "installed_capa", ConstantData(100))

    database.add_data(
        "CAND",
        "invest_cost",
        TreeData(
            {
                "parent": ConstantData(100),
                "child": ConstantData(300),
            }
        ),
    )

    database.add_data(
        "CAND",
        "max_invest",
        TreeData(
            {
                "parent": ConstantData(80),
                "child": ConstantData(100),
            }
        ),
    )

    # === Network ===
    demand_par = demand.replicate()
    candidate_par = candidate.replicate()

    network_par = Network("parent_test")
    network_par.add_node(node)
    network_par.add_component(demand_par)
    network_par.add_component(candidate_par)
    network_par.connect(
        PortRef(demand_par, "balance_port"), PortRef(node, "balance_port")
    )
    network_par.connect(
        PortRef(candidate_par, "balance_port"), PortRef(node, "balance_port")
    )

    demand_chd = demand.replicate()
    candidate_chd = candidate.replicate()

    network_chd = Network("child_test")
    network_chd.add_node(node)
    network_chd.add_component(demand_chd)
    network_chd.add_component(candidate_chd)
    network_chd.connect(
        PortRef(demand_chd, "balance_port"), PortRef(node, "balance_port")
    )
    network_chd.connect(
        PortRef(candidate_chd, "balance_port"), PortRef(node, "balance_port")
    )

    # === Decision tree creation ===
    config = InterDecisionTimeScenarioConfig([TimeBlock(0, [0])], 1)

    decision_tree_par = DecisionTreeNode("parent", config, network_par)
    decision_tree_chd = DecisionTreeNode(
        "child", config, network_chd, parent=decision_tree_par
    )

    # === Coupling model ===
    decision_tree_par.define_coupling_constraint(
        candidate_par,
        "invested_capa",
        candidate_par,
        "invested_capa",
        "delta_invest",
    )
    decision_tree_chd.define_coupling_constraint(
        candidate_par,
        "invested_capa",
        candidate_chd,
        "invested_capa",
        "delta_invest",
    )

    # === Build problem ===
    xpansion = build_benders_decomposed_problem(decision_tree_par, database)

    data = {
        "solution": {
            "overall_cost": 17_000,
            "values": {
                "parent_CAND_delta_invest": 80,
                "child_CAND_delta_invest": 20,
                "parent_CAND_invested_capa": 80,
                "child_CAND_invested_capa": 100,
            },
        }
    }
    solution = BendersSolution(data)

    # === Run ===
    assert xpansion.run()
    decomposed_solution = xpansion.solution
    if decomposed_solution is not None:  # For mypy only
        assert decomposed_solution.is_close(
            solution
        ), f"Solution differs from expected: {decomposed_solution}"


def test_investment_pathway_on_a_tree_with_one_root_two_children(
    generator: Component,
    candidate: Component,
    demand: Component,
    node: Node,
) -> None:
    """
    This use case aims at representing the situation where investment decisions are to be made at different, say "planning times".
    An actualization rate can be taken into account.

    The novelty compared the actual usage of planning tools, is that the planning decisions at a given time
    are taken without knowing exactly which "macro-scenario" / hypothesis on the system that will eventually happen
    (only knowing the probability distribution of these hypothesis).

    This example models a case where investment decisions have to be made in 2030 and 2040.
        - In 2030, we have full knowledge of the existing assets
        - In 2040, two possible hypothesis are possible :
            - P=0.2 => A case where there is no change in the generation assets since 2030 (except the potential investment in 2030)
            - P=0.8 => A case where a base generation unit is present

    When taking the decision in 2030, we do not know which case will occur in 2040
    and we seek the best decision given a risk criterion (the expectation here).
    The value of these models lies in the output for the first decision rather than
    the decisions at the later stages as the first decisions are related to "what we have to do today" ?
    More specifically, to define the use case, we define the following tree representing the system at the different decision times and hypothesis

    2030 (root node) :
        Demand = 300
        Generator :
            P_max = 200,
            Production cost = 10,
            Max investment = 400,
            Investment cost = 100
        Unsupplied energy :
            Cost = 10 000

    2040 with new base (child A) :
        Demand = 600
        Generator :
            P_max = 200,
            Production cost = 10,
            Max investment = 100,
            Investment cost = 50
        Base :
            P_max = 200,
            Production cost = 5
        Unsupplied energy :
            Cost = 10 000

    2040 no base (child B) :
        Demand = 600
        Generator :
            P_max = 200,
            Production cost = 10,
            Max investment = 100,
            Investment cost = 50
        Unsupplied energy :
            Cost = 10 000

    In the second decision time, demand increases from 300 to 600 in both scenarios.
    However, investment capacity in the candidate is limited to 100 in the second stage.
    Investment cost decreases to reflect the effect of a discount rate.

    In case 1, a base unit of capacity 100 has arrived and can produce at smaller cost than the candidate.
    As it is more interesting to invest the latest possible, the optimal solution for this scenario is to invest [100, 100].

    In case 2, there is no base unit and the max investment is 100 in the second stage,
    therefore if we consider scenario 2 only, as unsupplied energy is very expensive, the best investment is [300, 100]

    But here as we solve on the tree, we need to find the best solution in expectation on the set of paths in the tree.

    Case 1 :    prob    |    investment   |  operational
    root:             1 x [     100 x 100 +     10 x 300 ]
    child A:      + 0.8 x [      50 x 100 +     10 x 400 (generator) + 5 x 200 (base)]
    child B:      + 0.2 x [      50 x 100 +     10 x 400 (generator) + 10 000 x 200 (unsupplied energy)]
                  = 422 800

    Case 2 :    prob    |    investment   |  operational
    root:             1 x [     100 x 300 +     10 x 300 ]
    child A:      + 0.8 x [      50 x   0 +     10 x 400 (generator) + 5 x 200 (base)]
    child B:      + 0.2 x [      50 x 100 +     10 x 600 (generator)]
                  = 39 200

    As investing less than 300 in the first stage would increase the unsupplied energy and lead to an increase in overall cost
    (-1 MW invested in 1st stage => + 1 MW unsupplied energy => +900/MW cost increase more or less), the optimal solution is to invest :
        - 300 at first stage
        - 0 in child A
        - 100 in child B
    """

    # === Populating Database ===
    database = DataBase()
    database.add_data("N", "spillage_cost", ConstantData(10))
    database.add_data("N", "ens_cost", ConstantData(10_000))

    database.add_data(
        "D",
        "demand",
        TreeData(
            {
                "root": ConstantData(300),
                "childA": ConstantData(600),
                "childB": ConstantData(600),
            }
        ),
    )

    database.add_data("CAND", "op_cost", ConstantData(10))
    database.add_data("CAND", "installed_capa", ConstantData(200))
    database.add_data(
        "CAND",
        "invest_cost",
        TreeData(
            {
                "root": ConstantData(100),
                "childA": ConstantData(50),
                "childB": ConstantData(50),
            }
        ),
    )
    database.add_data(
        "CAND",
        "max_invest",
        TreeData(
            {
                "root": ConstantData(400),
                "childA": ConstantData(100),
                "childB": ConstantData(100),
            }
        ),
    )

    database.add_data("BASE", "p_max", ConstantData(200))
    database.add_data("BASE", "cost", ConstantData(5))

    # === Network ===
    network_root = Network("root_network")
    network_root.add_node(node)
    network_root.add_component(demand)
    network_root.add_component(candidate)
    network_root.connect(PortRef(demand, "balance_port"), PortRef(node, "balance_port"))
    network_root.connect(
        PortRef(candidate, "balance_port"), PortRef(node, "balance_port")
    )

    network_childA = network_root.replicate(id="childA_network")
    network_childA.add_component(generator)
    network_childA.connect(
        PortRef(generator, "balance_port"), PortRef(node, "balance_port")
    )

    network_childB = network_root.replicate(id="childB_network")

    # === Decision tree creation ===
    time_scenario_config = InterDecisionTimeScenarioConfig([TimeBlock(0, [0])], 1)

    dt_root = DecisionTreeNode("root", time_scenario_config, network_root)
    dt_child_A = DecisionTreeNode(
        "childA", time_scenario_config, network_childA, parent=dt_root, prob=0.8
    )
    dt_child_B = DecisionTreeNode(
        "childB", time_scenario_config, network_childB, parent=dt_root, prob=0.2
    )

    # === Coupling model ===
    dt_root.define_coupling_constraint(
        candidate,
        "invested_capa",
        candidate,
        "invested_capa",
        "delta_invest",
    )
    dt_child_A.define_coupling_constraint(
        candidate,
        "invested_capa",
        candidate,
        "invested_capa",
        "delta_invest",
    )
    dt_child_B.define_coupling_constraint(
        candidate,
        "invested_capa",
        candidate,
        "invested_capa",
        "delta_invest",
    )

    # === Build problem ===
    xpansion = build_benders_decomposed_problem(dt_root, database)

    data = {
        "solution": {
            "overall_cost": 39_200,
            "values": {
                "root_CAND_delta_invest": 300,
                "childA_CAND_delta_invest": 0,
                "childB_CAND_delta_invest": 100,
                "root_CAND_invested_capa": 300,
                "childA_CAND_invested_capa": 300,
                "childB_CAND_invested_capa": 400,
            },
        }
    }
    solution = BendersSolution(data)

    # === Run ===
    assert xpansion.run()
    decomposed_solution = xpansion.solution
    if decomposed_solution is not None:  # For mypy only
        assert decomposed_solution.is_close(
            solution
        ), f"Solution differs from expected: {decomposed_solution}"
