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


def test_extra_output_abs_round_on_variable() -> None:
    """
    abs() and round() applied to a decision variable are allowed in extra
    outputs (post-solve evaluation), even though they would be rejected as
    nonlinear inside a constraint or bound.
    """
    from gems.expression import var
    from gems.expression.expression import literal
    from gems.model.model import model
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import DataBase, Study, System, create_component

    SIMPLE_MODEL = model(
        id="SIMPLE_ABS_ROUND",
        variables=[
            float_variable("a", lower_bound=literal(2.7), upper_bound=literal(2.7))
        ],
        extra_outputs={
            "abs_shift": (var("a") - literal(5)).abs(),
            "rounded": var("a").round(),
        },
    )

    database = DataBase()
    comp = create_component(model=SIMPLE_MODEL, id="comp_1")

    system = System("test_abs_round_extra")
    system.add_component(comp)

    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), scenario_ids=list(range(1))
    )
    problem.solve(solver_name="highs")

    df = SimulationTableBuilder().build(problem)
    abs_shift = (
        df.component("comp_1").output("abs_shift").value(time_index=0, scenario_index=0)
    )
    rounded = (
        df.component("comp_1").output("rounded").value(time_index=0, scenario_index=0)
    )
    assert abs_shift == pytest.approx(2.3)
    assert rounded == pytest.approx(3.0)


def test_extra_output_min_on_variable() -> None:
    """
    min(variable, parameter) is allowed in extra outputs and evaluated
    element-wise post-solve (issue #237).
    """
    from gems.expression import param, var
    from gems.expression.expression import literal, minimum
    from gems.model.model import model
    from gems.model.parameter import float_parameter
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import ConstantData, DataBase, Study, System, create_component

    SIMPLE_MODEL = model(
        id="SIMPLE_MIN",
        parameters=[float_parameter("cap")],
        variables=[float_variable("a", lower_bound=literal(5), upper_bound=literal(5))],
        extra_outputs={"capped": minimum(var("a"), param("cap"))},
    )

    database = DataBase()
    comp = create_component(model=SIMPLE_MODEL, id="comp_1")
    database.add_data("comp_1", "cap", ConstantData(3.0))

    system = System("test_min_extra")
    system.add_component(comp)

    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), scenario_ids=list(range(1))
    )
    problem.solve(solver_name="highs")

    df = SimulationTableBuilder().build(problem)
    capped = (
        df.component("comp_1").output("capped").value(time_index=0, scenario_index=0)
    )
    # min(a=5, cap=3) == 3
    assert capped == pytest.approx(3.0)


def test_extra_output_comparison() -> None:
    """
    Comparison operators (>=, <=) are allowed in extra outputs and evaluated
    post-solve as a float indicator: 1.0 where the condition holds, 0.0
    otherwise (issue #237). Previously these raised NotImplementedError.
    """
    from gems.expression import param, var
    from gems.expression.expression import Comparator, ComparisonNode, literal
    from gems.model.model import model
    from gems.model.parameter import float_parameter
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import ConstantData, DataBase, Study, System, create_component

    SIMPLE_MODEL = model(
        id="SIMPLE_CMP",
        parameters=[float_parameter("threshold")],
        variables=[float_variable("a", lower_bound=literal(5), upper_bound=literal(5))],
        extra_outputs={
            # a (=5) >= threshold (=3) -> 1.0
            "ge": ComparisonNode(var("a"), param("threshold"), Comparator.GREATER_THAN),
            # a (=5) <= threshold (=3) -> 0.0
            "le": ComparisonNode(var("a"), param("threshold"), Comparator.LESS_THAN),
        },
    )

    database = DataBase()
    comp = create_component(model=SIMPLE_MODEL, id="comp_1")
    database.add_data("comp_1", "threshold", ConstantData(3.0))

    system = System("test_comparison_extra")
    system.add_component(comp)

    problem = build_problem(
        Study(system, database), TimeBlock(1, [0]), scenario_ids=list(range(1))
    )
    problem.solve(solver_name="highs")

    df = SimulationTableBuilder().build(problem)
    ge = df.component("comp_1").output("ge").value(time_index=0, scenario_index=0)
    le = df.component("comp_1").output("le").value(time_index=0, scenario_index=0)
    assert ge == pytest.approx(1.0)
    assert le == pytest.approx(0.0)
