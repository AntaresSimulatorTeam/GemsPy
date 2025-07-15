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

from unittest.mock import Mock

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

    mock_variable_component.solution_value.side_effect = lambda: 1.0

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

    mock_problem.context = opt_context
    output = OutputValues(mock_problem)

    test_output = OutputValues()
    assert output != test_output, f"Output is equal to empty output: {output}"

    test_output.component("component_id_test").ignore = True
    assert (
        output == test_output
    ), f"Output differs from the expected output after 'ignore': {output}"

    test_output.component("component_id_test").ignore = False
    test_output.component("component_id_test").var("component_var_name").value = 1.0
    test_output.component("component_id_test").var(
        "component_approx_var_name"
    ).ignore = True

    assert (
        output == test_output
    ), f"Output differs from the expected after 'var_name': {output}"

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
    ), f"Output differs from the expected inside tolerance: {output}"

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
