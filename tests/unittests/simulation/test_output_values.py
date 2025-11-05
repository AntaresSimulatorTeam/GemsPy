# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import Mock, patch

import ortools.linear_solver.pywraplp as lp

from gems.simulation.optimization import (
    OptimizationContext,
    OptimizationProblem,
    TimestepComponentVariableKey,
)
from gems.simulation.output_values import OutputComponent, OutputValues


def test_component_and_flow_output_object() -> None:
    mock_variable_component = Mock(spec=lp.Variable)
    mock_variable_component.solution_value.side_effect = lambda: 1.0

    opt_context = Mock(spec=OptimizationContext)
    opt_context.get_all_component_variables.return_value = {
        TimestepComponentVariableKey(
            component_id="component_id_test",
            variable_name="component_var_name",
            block_timestep=0,
            scenario=0,
        ): mock_variable_component,
        TimestepComponentVariableKey(
            component_id="component_id_test",
            variable_name="component_approx_var_name",
            block_timestep=0,
            scenario=0,
        ): mock_variable_component,
    }
    opt_context.block_length.return_value = 1
    opt_context.network = Mock()
    opt_context.network.all_components = []

    mock_solver = Mock()
    mock_solver.IsMip.return_value = False

    mock_problem = Mock(spec=OptimizationProblem)
    mock_problem.context = opt_context
    mock_problem.solver = mock_solver

    with patch.object(
        OutputComponent,
        "evaluate_extra_outputs",
        return_value={},
    ):
        actual_output = OutputValues(mock_problem)

    expected_output = OutputValues()
    assert actual_output != expected_output

    expected_output.component("component_id_test").ignore = True
    assert actual_output == expected_output

    expected_output.component("component_id_test").ignore = False
    expected_output.component("component_id_test").var("component_var_name").value = 1.0
    expected_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True
    assert actual_output == expected_output

    expected_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = False
    expected_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_001
    assert actual_output != expected_output and not actual_output.is_close(
        expected_output
    )

    expected_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_000_1
    assert actual_output != expected_output and actual_output.is_close(expected_output)

    expected_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True
    expected_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).value = 1.0
    assert actual_output != expected_output

    expected_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).ignore = True
    assert actual_output == expected_output

    print(actual_output)
