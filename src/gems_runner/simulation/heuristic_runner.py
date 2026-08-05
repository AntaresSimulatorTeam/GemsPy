# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

"""Two-pass thermal heuristic solving.

After a first LP solve (with continuous relaxation for HEURISTIC-strategy
components), this module computes heuristic bounds per component and scenario
and injects them as additional linopy constraints before a second solve.
"""

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import xarray as xr

from gems_craft.optim_config.parsing import (
    HeuristicElementConfig,
    ModelElementAccessType,
    OptimConfig,
    get_heuristic_config_map,
)
from gems_craft.study.parsing import HeuristicId, IntegerStrategyId
from gems_runner.simulation.thermal_heuristic import (
    find_min_generation_fast,
    find_num_units_accurate,
)
from linopy import Variable

if TYPE_CHECKING:
    from gems_craft.study.study import Study
    from gems_runner.simulation.optimization import OptimizationProblem

_HEURISTIC_FUNCTIONS: Dict[HeuristicId, Callable[..., List]] = {
    HeuristicId.FAST: find_min_generation_fast,
    HeuristicId.ACCURATE: find_num_units_accurate,
}


def should_apply_heuristics(study: "Study") -> bool:
    """Return True when the two-pass heuristic solve is needed."""
    return any(
        c.integer_strategy.id == IntegerStrategyId.HEURISTIC
        for c in study.system.all_components
    )


def _get_component_linopy_var(
    problem: "OptimizationProblem", model_id: str, name: str, component_id: str
) -> Variable:
    """Like ``_get_linopy_var``, but returns the actually-registered Variable for
    *component_id* rather than the (possibly relaxed/exact-merged) one in
    ``problem._linopy_vars`` — bound mutations only reach the solver through this one.
    """
    linopy_var = problem.get_component_variable(model_id, name, component_id)
    if linopy_var is None:
        raise ValueError(
            f"Variable '{name}' not found in model '{model_id}' for component '{component_id}'."
        )
    return linopy_var


def _read_da(
    da: "xr.DataArray", component_id: str, local_scenario_idx: int
) -> Union[float, int, List[float], List[int]]:
    arr = da.sel(component=component_id)
    if "scenario" in arr.dims:
        arr = arr.isel(scenario=local_scenario_idx)
    if "time" in arr.dims:
        return arr.values.tolist()
    return float(arr.item())


def _resolve_input_for_heuristic(
    problem: "OptimizationProblem",
    model_id: str,
    input_config: HeuristicElementConfig,
    component_id: str,
    local_scenario_idx: int,
) -> Union[float, int, List[float], List[int]]:
    name = input_config.id
    access = input_config.type

    if access == ModelElementAccessType.VARIABLE_SOLUTION:
        if problem.linopy_model.solution is None:
            raise RuntimeError("Problem must be solved before applying heuristics.")
        sol_da = problem.get_variable_solution(model_id, name)
        if sol_da is None:
            raise ValueError(f"Variable '{name}' not found in model '{model_id}'.")
        return _read_da(sol_da, component_id, local_scenario_idx)
    elif access == ModelElementAccessType.VARIABLE_LOWER_BOUND:
        return _read_da(
            _get_component_linopy_var(problem, model_id, name, component_id).lower,
            component_id,
            local_scenario_idx,
        )
    elif access == ModelElementAccessType.VARIABLE_UPPER_BOUND:
        return _read_da(
            _get_component_linopy_var(problem, model_id, name, component_id).upper,
            component_id,
            local_scenario_idx,
        )
    elif access == ModelElementAccessType.PARAMETER:
        param_arr = problem.param_arrays.get((model_id, name))
        if param_arr is None:
            raise ValueError(f"Parameter '{name}' not found in model '{model_id}'.")
        return _read_da(param_arr, component_id, local_scenario_idx)
    else:
        raise ValueError(
            f"Unknown access type '{access}' for heuristic input '{name}'."
        )


def _apply_heuristic_bounds(
    linopy_var: Any,
    result_da: "xr.DataArray",
    access: ModelElementAccessType,
    component_id: str,
    local_scenario_idx: int,
) -> None:
    has_scenario_dim = "scenario" in linopy_var.dims

    if access == ModelElementAccessType.VARIABLE_LOWER_BOUND:
        bound = linopy_var.lower
    elif access == ModelElementAccessType.VARIABLE_UPPER_BOUND:
        bound = linopy_var.upper
    else:
        raise ValueError(
            f"Invalid output type '{access}' — only variable bounds are allowed."
        )

    comp_selector = bound.sel(component=component_id)
    if has_scenario_dim:
        comp_selector = comp_selector.isel(scenario=local_scenario_idx)
    comp_selector[:] = result_da


def apply_thermal_heuristics(
    problem: "OptimizationProblem",
    optim_config: OptimConfig,
    scenario_ids: List[int],
) -> None:
    """Update variable bounds in *problem* after a first LP solve using heuristics.

    For each component with ``integer_strategy=HEURISTIC``, the heuristic function
    is called per scenario and the resulting values are applied directly as lower
    or upper bounds on the target variable.  A second ``solve()`` then enforces them.

    Parameters
    ----------
    problem:
        Solved OptimizationProblem whose LP solution drives the heuristics.
    optim_config:
        Parsed OptimConfig containing the heuristic configurations.
    scenario_ids:
        The list of MC scenario indices used when building *problem*.
    """
    heuristic_config_map = get_heuristic_config_map(optim_config)
    solver_name = optim_config.solver_options.name

    heuristic_comps = [
        c
        for c in problem.study.system.all_components
        if c.integer_strategy.id == IntegerStrategyId.HEURISTIC
        and c.integer_strategy.heuristic_id is not None
    ]

    for component in heuristic_comps:
        model_id = component.model.id
        heuristic_id = component.integer_strategy.heuristic_id
        assert heuristic_id is not None  # for mypy
        heuristic_config = heuristic_config_map[(model_id, heuristic_id.value)]
        heuristic_fn = _HEURISTIC_FUNCTIONS[heuristic_config.id]
        fn_params = inspect.signature(heuristic_fn).parameters

        for local_idx in range(len(scenario_ids)):
            kwargs = {
                inp.heuristic_element: _resolve_input_for_heuristic(
                    problem,
                    model_id,
                    inp,
                    component.id,
                    local_idx,
                )
                for inp in heuristic_config.inputs
            }
            if "solver_name" in fn_params:
                kwargs["solver_name"] = solver_name
            heuristic_result: List = heuristic_fn(**kwargs)  # type: ignore[call-overload]

            result_da = xr.DataArray(
                heuristic_result,
                dims=["time"],
                coords={"time": list(range(len(heuristic_result)))},
            )

            for output in heuristic_config.outputs:
                linopy_var = _get_component_linopy_var(
                    problem, model_id, output.id, component.id
                )

                _apply_heuristic_bounds(
                    linopy_var, result_da, output.type, component.id, local_idx
                )
