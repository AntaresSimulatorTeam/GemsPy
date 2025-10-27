# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import Mock, patch

import ortools.linear_solver.pywraplp as lp

from gems.simulation import OutputValues
from gems.simulation.extra_output import evaluate_extra_outputs_for_a_component
from gems.simulation.optimization import (
    OptimizationContext,
    OptimizationProblem,
    TimestepComponentVariableKey,
)


def test_component_and_flow_output_object() -> None:
    # -------------------------------
    # Setup mock problem and context
    # -------------------------------
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

    # ---------------------------------------------
    # Patch extra output evaluation to return empty
    # ---------------------------------------------
    with patch(
        "gems.simulation.extra_output.evaluate_extra_outputs_for_a_component",
        return_value={},
    ):
        output = OutputValues(mock_problem)

    # ------------------------
    # Build expected OutputValues
    # ------------------------
    test_output = OutputValues()
    assert (
        output != test_output
    ), "Output should not equal an empty OutputValues initially"

    # Ignore component and compare
    test_output.component("component_id_test").ignore = True
    assert (
        output == test_output
    ), "Output should equal test_output after ignoring component"

    # Set variable values and ignore others
    test_output.component("component_id_test").ignore = False
    test_output.component("component_id_test").var("component_var_name").value = 1.0
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True
    assert (
        output == test_output
    ), "Output should match after setting variable values and ignoring others"

    # Test values outside tolerance
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = False
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_001
    assert output != test_output and not output.is_close(
        test_output
    ), "Output should differ outside tolerance"

    # Test values inside tolerance
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).value = 1.000_000_000_1
    assert output != test_output and output.is_close(
        test_output
    ), "Output should match inside tolerance"

    # Add wrong variable and ignore it
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True
    test_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).value = 1.0
    assert output != test_output, "Output should differ with wrong variable"

    test_output.component("component_id_test").var(
        "wrong_component_var_name"
    ).ignore = True
    assert output == test_output, "Output should match after ignoring wrong variable"

    # Print final output
    print(output)
