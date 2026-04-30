# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

import pytest

from gems.simulation.simulation_table import SimulationTableBuilder


def test_extra_output_with_sum_connections() -> None:
    """
    Extra output using sum_connections is evaluated correctly via
    VectorizedExtraOutputBuilder + build_port_arrays.

    Setup: gen_1 (GEN model, variable gen=5) connects to node_1 (NODE model).
    NODE model has extra output 'total_flow' = sum_connections(balance_port.flow).
    GEN model defines balance_port.flow = var("gen").
    Expected: total_flow at node_1 == 5.0.
    """
    from gems.expression import var
    from gems.expression.expression import literal, port_field
    from gems.model.model import ModelPort, model
    from gems.model.port import PortField, PortFieldDefinition, PortFieldId, PortType
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import Component, DataBase, PortRef, Study, System, create_component

    BALANCE_PORT_TYPE = PortType(id="balance", fields=[PortField("flow")])

    # Generator: variable gen fixed to 5, exposes it as port flow.
    GEN_MODEL = model(
        id="GEN_EXTRA",
        variables=[
            float_variable("gen", lower_bound=literal(5), upper_bound=literal(5))
        ],
        ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId("balance_port", "flow"),
                definition=var("gen"),
            )
        ],
    )

    # Node: slave port, extra output = sum of incoming flows (no binding constraint).
    NODE_MODEL = model(
        id="NODE_EXTRA",
        ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
        extra_outputs={
            "total_flow": port_field("balance_port", "flow").sum_connections()
        },
    )

    database = DataBase()

    gen_comp = create_component(model=GEN_MODEL, id="gen_1")
    node_comp = Component(model=NODE_MODEL, id="node_1")

    system = System("test_sum_connections")
    system.add_component(gen_comp)
    system.add_component(node_comp)
    system.connect(
        PortRef(gen_comp, "balance_port"), PortRef(node_comp, "balance_port")
    )

    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), scenario_ids=list(range(1))
    )
    problem.solve(solver_name="highs")

    df = SimulationTableBuilder().build(problem)
    total_flow = (
        df.component("node_1")
        .output("total_flow")
        .value(time_index=0, scenario_index=0)
    )
    assert total_flow == pytest.approx(5.0)


def test_extra_output_nonlinear() -> None:
    """
    Nonlinear extra output (var * var) is correctly evaluated.

    VectorizedExtraOutputBuilder allows products of variables since extra
    outputs are not solver constraints. Equivalent VectorizedLinearExprBuilder
    would raise NotImplementedError for the same expression.

    Setup: one component with variable a=3 (fixed). Extra output squared = a*a.
    Expected: squared = 9.0.
    """
    from gems.expression import var
    from gems.expression.expression import literal
    from gems.model.model import model
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import DataBase, Study, System, create_component

    SIMPLE_MODEL = model(
        id="SIMPLE_NL",
        variables=[float_variable("a", lower_bound=literal(3), upper_bound=literal(3))],
        extra_outputs={"squared": var("a") * var("a")},
    )

    database = DataBase()
    comp = create_component(model=SIMPLE_MODEL, id="comp_1")

    system = System("test_nonlinear")
    system.add_component(comp)

    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), scenario_ids=list(range(1))
    )
    problem.solve(solver_name="highs")

    df = SimulationTableBuilder().build(problem)
    squared = (
        df.component("comp_1").output("squared").value(time_index=0, scenario_index=0)
    )
    assert squared == pytest.approx(9.0)
