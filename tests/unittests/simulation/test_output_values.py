# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import Mock, patch

import ortools.linear_solver.pywraplp as lp

from gems.simulation import OutputValues
from gems.simulation.optimization import (
    OptimizationContext,
    OptimizationProblem,
    TimestepComponentVariableKey,
)


def test_component_and_flow_output_object() -> None:
    mock_variable_component = Mock(spec=lp.Variable)
    mock_problem = Mock(spec=OptimizationProblem)
    opt_context = Mock(spec=OptimizationContext)

    # Fake solver variable value
    mock_variable_component.solution_value.side_effect = lambda: 1.0

    # Fake component variable mapping
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

    mock_problem.context = opt_context
    mock_solver = Mock()
    mock_solver.IsMip.return_value = False
    mock_problem.solver = mock_solver

    with patch(
        "gems.simulation.output_values.evaluate_all_extra_outputs", return_value={}
    ):
        output = OutputValues(mock_problem)

    test_output = OutputValues()
    assert output != test_output, f"Output is equal to empty output: {output}"

    test_output.component("component_id_test").ignore = True
    assert (
        output == test_output
    ), f"Output differs from expected output after 'ignore': {output}"

    test_output.component("component_id_test").ignore = False
    test_output.component("component_id_test").var("component_var_name").value = 1.0
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True

    assert (
        output == test_output
    ), f"Output differs from expected output after setting variable values: {output}"

    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = False
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_001

    assert output != test_output and not output.is_close(
        test_output
    ), f"Output is equal to expected outside tolerance: {output}"

    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_000_1

    assert output != test_output and output.is_close(
        test_output
    ), f"Output differs from expected inside tolerance: {output}"

    # Add extra wrong variable and ignore it
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True
    test_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).value = 1.0

    assert output != test_output, f"Output is equal to wrong output: {output}"

    test_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).ignore = True

    assert output == test_output, f"Output differs from expected: {output}"

    print(output)
