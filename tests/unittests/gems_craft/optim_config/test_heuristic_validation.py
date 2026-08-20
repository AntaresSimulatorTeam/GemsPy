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

import pytest

from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model.model import Model, model
from gems_craft.model.parameter import float_parameter
from gems_craft.model.variable import float_variable
from gems_craft.optim_config.parsing import (
    HeuristicConfig,
    HeuristicElementConfig,
    ModelElementAccessType,
    ModelOptimConfig,
    OptimConfig,
)
from gems_craft.optim_config.validation import validate_optim_config
from gems_craft.study.parsing import HeuristicId
from gems_craft.study.system import System, create_component


def _structure(time_dependent: bool) -> IndexingStructure:
    return IndexingStructure(time=time_dependent, scenario=True)


def _thermal_model(
    *,
    min_up_duration_time_dependent: bool = False,
    min_down_duration_time_dependent: bool = False,
    generation_power_time_dependent: bool = True,
    cluster_max_generation_time_dependent: bool = True,
    num_units_on_time_dependent: bool = True,
    num_units_max_param_time_dependent: bool = True,
) -> Model:
    return model(
        id="thermal",
        parameters=[
            float_parameter(
                "min_up_duration", _structure(min_up_duration_time_dependent)
            ),
            float_parameter(
                "min_down_duration", _structure(min_down_duration_time_dependent)
            ),
            float_parameter("min_power_per_unit", _structure(False)),
            float_parameter("max_power_per_unit", _structure(False)),
            float_parameter(
                "cluster_max_generation",
                _structure(cluster_max_generation_time_dependent),
            ),
            float_parameter(
                "num_units_max_param", _structure(num_units_max_param_time_dependent)
            ),
        ],
        variables=[
            float_variable(
                "generation_power",
                structure=_structure(generation_power_time_dependent),
            ),
            float_variable(
                "num_units_on", structure=_structure(num_units_on_time_dependent)
            ),
        ],
    )


def _fast_heuristic_config(
    cluster_max_generation_id: str = "cluster_max_generation",
) -> HeuristicConfig:
    return HeuristicConfig(
        id=HeuristicId.FAST,
        inputs=[
            HeuristicElementConfig(
                heuristic_element="generation_power",
                id="generation_power",
                type=ModelElementAccessType.VARIABLE_SOLUTION,
            ),
            HeuristicElementConfig(
                heuristic_element="cluster_max_generation",
                id=cluster_max_generation_id,
            ),
            HeuristicElementConfig(
                heuristic_element="min_power_per_unit", id="min_power_per_unit"
            ),
            HeuristicElementConfig(
                heuristic_element="max_power_per_unit", id="max_power_per_unit"
            ),
            HeuristicElementConfig(
                heuristic_element="min_up_duration", id="min_up_duration"
            ),
            HeuristicElementConfig(
                heuristic_element="min_down_duration", id="min_down_duration"
            ),
        ],
        outputs=[
            HeuristicElementConfig(
                heuristic_element="minimum_generation_power",
                id="generation_power",
                type=ModelElementAccessType.VARIABLE_LOWER_BOUND,
            ),
        ],
    )


def _accurate_heuristic_config(
    num_units_max_id: str = "num_units_max_param",
    num_units_max_type: ModelElementAccessType = ModelElementAccessType.PARAMETER,
) -> HeuristicConfig:
    return HeuristicConfig(
        id=HeuristicId.ACCURATE,
        inputs=[
            HeuristicElementConfig(
                heuristic_element="num_units_on_opt",
                id="num_units_on",
                type=ModelElementAccessType.VARIABLE_SOLUTION,
            ),
            HeuristicElementConfig(
                heuristic_element="num_units_max",
                id=num_units_max_id,
                type=num_units_max_type,
            ),
            HeuristicElementConfig(
                heuristic_element="min_up_duration", id="min_up_duration"
            ),
            HeuristicElementConfig(
                heuristic_element="min_down_duration", id="min_down_duration"
            ),
        ],
        outputs=[
            HeuristicElementConfig(
                heuristic_element="minimum_num_units_on",
                id="num_units_on",
                type=ModelElementAccessType.VARIABLE_LOWER_BOUND,
            ),
        ],
    )


def _validate(model: Model, heuristic_config: HeuristicConfig) -> None:
    config = OptimConfig(
        models=[ModelOptimConfig(id="thermal", heuristics=[heuristic_config])]
    )
    system = System(id="test")
    system.add_component(create_component(model=model, id="cluster1"))
    validate_optim_config(config, system)


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------


def test_fast_heuristic_default_structures_pass() -> None:
    _validate(_thermal_model(), _fast_heuristic_config())


def test_accurate_heuristic_default_structures_pass() -> None:
    _validate(_thermal_model(), _accurate_heuristic_config())


def test_cluster_max_generation_accepts_time_dependent() -> None:
    model = _thermal_model(cluster_max_generation_time_dependent=True)
    _validate(model, _fast_heuristic_config())


def test_cluster_max_generation_accepts_time_independent() -> None:
    model = _thermal_model(cluster_max_generation_time_dependent=False)
    _validate(model, _fast_heuristic_config())


def test_num_units_max_accepts_time_dependent() -> None:
    model = _thermal_model(num_units_max_param_time_dependent=True)
    _validate(model, _accurate_heuristic_config())


def test_num_units_max_accepts_time_independent() -> None:
    model = _thermal_model(num_units_max_param_time_dependent=False)
    _validate(model, _accurate_heuristic_config())


# ---------------------------------------------------------------------------
# Time-dependence mismatches
# ---------------------------------------------------------------------------


def test_min_up_duration_wrongly_time_dependent_raises() -> None:
    model = _thermal_model(min_up_duration_time_dependent=True)
    with pytest.raises(
        ValueError, match="min_up_duration.*must be 'time-dependent:False'"
    ):
        _validate(model, _accurate_heuristic_config())


def test_min_down_duration_wrongly_time_dependent_raises() -> None:
    model = _thermal_model(min_down_duration_time_dependent=True)
    with pytest.raises(
        ValueError, match="min_down_duration.*must be 'time-dependent:False'"
    ):
        _validate(model, _accurate_heuristic_config())


def test_generation_power_wrongly_time_independent_raises() -> None:
    model = _thermal_model(generation_power_time_dependent=False)
    with pytest.raises(
        ValueError, match="generation_power.*must be 'time-dependent:True'"
    ):
        _validate(model, _fast_heuristic_config())


def test_num_units_on_opt_wrongly_time_independent_raises() -> None:
    model = _thermal_model(num_units_on_time_dependent=False)
    with pytest.raises(
        ValueError, match="num_units_on_opt.*must be 'time-dependent:True'"
    ):
        _validate(model, _accurate_heuristic_config())


# ---------------------------------------------------------------------------
# Existence / kind mismatches
# ---------------------------------------------------------------------------


def test_unknown_parameter_id_raises() -> None:
    model = _thermal_model()
    heuristic_config = _fast_heuristic_config(
        cluster_max_generation_id="does_not_exist"
    )
    with pytest.raises(ValueError, match="'does_not_exist' .* not found in model"):
        _validate(model, heuristic_config)


def test_id_of_wrong_kind_raises() -> None:
    # 'min_power_per_unit' exists as a parameter, not as a variable.
    model = _thermal_model()
    heuristic_config = _accurate_heuristic_config(
        num_units_max_id="min_power_per_unit",
        num_units_max_type=ModelElementAccessType.VARIABLE_UPPER_BOUND,
    )
    with pytest.raises(ValueError, match="'min_power_per_unit' .* not found in model"):
        _validate(model, heuristic_config)
