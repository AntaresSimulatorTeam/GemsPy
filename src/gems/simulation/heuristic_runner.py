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

"""Two-pass thermal heuristic solving.

After a first LP solve (with continuous relaxation for HEURISTIC-strategy
components), this module computes heuristic bounds per component and scenario
and injects them as additional linopy constraints before a second solve.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import xarray as xr

from gems.optim_config.parsing import (
    HeuristicConfig,
    HeuristicElementConfig,
    ModelElementAccessType,
    OptimConfig,
)
from gems.simulation.thermal_heuristic import (
    find_min_generation_fast,
    find_nb_units_accurate,
)
from gems.study.parsing import HeuristicId, IntegerStrategyId

if TYPE_CHECKING:
    from gems.simulation.optimization import OptimizationProblem
    from gems.study.study import Study

_HEURISTIC_FUNCTIONS: Dict[HeuristicId, Callable[..., List]] = {
    HeuristicId.FAST: find_min_generation_fast,
    HeuristicId.ACCURATE: find_nb_units_accurate,
}


def should_apply_heuristics(study: "Study") -> bool:
    """Return True when the two-pass heuristic solve is needed."""
    return any(
        c.integer_strategy.id == IntegerStrategyId.HEURISTIC
        for c in study.system.all_components
    )


def _get_linopy_var(problem: "OptimizationProblem", model_id: str, name: str) -> Any:
    linopy_var = problem._linopy_vars.get((model_id, name))
    if linopy_var is None:
        raise ValueError(f"Variable '{name}' not found in model '{model_id}'.")
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
        linopy_var = _get_linopy_var(problem, model_id, name)
        solution = problem.linopy_model.solution
        if solution is None:
            raise RuntimeError("Problem must be solved before applying heuristics.")
        return _read_da(solution[linopy_var.name], component_id, local_scenario_idx)
    elif access == ModelElementAccessType.VARIABLE_LOWER_BOUND:
        return _read_da(
            _get_linopy_var(problem, model_id, name).lower,
            component_id,
            local_scenario_idx,
        )
    elif access == ModelElementAccessType.VARIABLE_UPPER_BOUND:
        return _read_da(
            _get_linopy_var(problem, model_id, name).upper,
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
    heuristic_config_map: Dict[str, HeuristicConfig] = {
        f"{mc.id}/{heuristic_config.id.value}": heuristic_config
        for mc in optim_config.models
        for heuristic_config in (mc.heuristic or [])
    }

    heuristic_comps = [
        c
        for c in problem.study.system.all_components
        if c.integer_strategy.id == IntegerStrategyId.HEURISTIC
        and c.integer_strategy.heuristic_id is not None
    ]

    for component in heuristic_comps:
        model_id = component.model.id
        heuristic_id = component.integer_strategy.heuristic_id
        assert heuristic_id is not None
        key = f"{model_id}/{heuristic_id.value}"
        if key not in heuristic_config_map:
            raise ValueError(
                f"Component '{component.id}' references heuristic "
                f"'{heuristic_id.value}' on model '{model_id}', "
                f"but this heuristic is not declared in optim-config."
            )
        heuristic_config = heuristic_config_map[key]
        heuristic_fn = _HEURISTIC_FUNCTIONS[heuristic_config.id]

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
            heuristic_result: List = heuristic_fn(**kwargs)  # type: ignore[call-overload]

            result_da = xr.DataArray(
                heuristic_result,
                dims=["time"],
                coords={"time": list(range(len(heuristic_result)))},
            )

            for output in heuristic_config.outputs:
                linopy_var = problem._linopy_vars.get((model_id, output.id))
                if linopy_var is None:
                    raise ValueError(
                        f"Heuristic output variable '{output.id}' not found "
                        f"in model '{model_id}'."
                    )

                _apply_heuristic_bounds(
                    linopy_var, result_da, output.type, component.id, local_idx
                )
