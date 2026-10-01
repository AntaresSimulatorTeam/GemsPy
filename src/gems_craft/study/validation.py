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

"""Cross-validation of a resolved system against the models and data it refers to.

Kept apart from `resolve_components.py` (which only resolves the parsed system
into its runtime objects) and from `study.py` (which only holds them together)
— mirroring `optim_config/parsing.py` and `optim_config/validation.py`.
"""

from typing import Dict, Set

from gems_craft.model import Model
from gems_craft.model.library import Library
from gems_craft.study.parsing import SetInstanceSchema, SystemSchema
from gems_craft.study.study import Study
from gems_craft.study.system import System


def check_component_models(system: System, input_models: Dict[str, Model]) -> bool:
    """
    Checks if all components in the System have a valid model from the library.
    Returns True if all components are consistent, raises ValueError otherwise.
    """
    # TODO: Update this check to verify that each component has a valid model from the lib it refers to (and not all libs)
    model_ids_set = input_models.keys()
    for component in system.all_components:
        if component.model.id not in model_ids_set:
            raise ValueError(
                f"Error: Component {component.id} has invalid model ID: {component.model.id}"
            )
    return True


def _check_set_elements(instance: SetInstanceSchema) -> None:
    elements = instance.elements
    if len(set(elements)) != len(elements):
        raise ValueError(f"Set '{instance.id}' has duplicate elements.")


def _sets_used_by_model(model: Model) -> Set[str]:
    """Every set id actually referenced via `indexed_by` by some parameter or
    variable of this model (local or global -- not distinguished here)."""
    used: Set[str] = set()
    for p in model.parameters.values():
        used |= p.structure.sets
    for v in model.variables.values():
        used |= v.structure.sets
    return used


def check_custom_sets(
    input_study: SystemSchema,
    model_dict: Dict[str, Model],
    lib_dict: Dict[str, Library],
) -> None:
    """
    Cross-validates declared global/local custom sets against their
    instantiation in system.yml:
      - every global set actually used by some component's model is
        instantiated at the system level;
      - every local set actually used by a component's model is instantiated
        by that component;
      - every instantiated set's element list is duplicate-free (an empty
        list is a valid, if degenerate, set).
    """
    used_models = [
        model_dict[c.model] for c in input_study.components if c.model in model_dict
    ]

    used_global_sets: Set[str] = set()
    for model in used_models:
        used_global_sets |= _sets_used_by_model(model) - model.local_sets

    instantiated_global = {s.id: s for s in (input_study.sets or [])}
    missing_global = used_global_sets - instantiated_global.keys()
    if missing_global:
        raise ValueError(
            f"Global set(s) {sorted(missing_global)} are used (via indexed_by) "
            "by a component's model but not instantiated in system.yml's "
            "system-level 'sets:'."
        )
    for s in instantiated_global.values():
        _check_set_elements(s)

    for component in input_study.components:
        component_model = model_dict.get(component.model)
        if component_model is None:
            continue  # reported by check_component_models
        used_local_sets = (
            _sets_used_by_model(component_model) & component_model.local_sets
        )
        instantiated_local = {s.id: s for s in (component.sets or [])}
        missing_local = used_local_sets - instantiated_local.keys()
        if missing_local:
            raise ValueError(
                f"Component '{component.id}' is missing instantiation for local "
                f"set(s) {sorted(missing_local)} used (via indexed_by) by its model."
            )
        for s in instantiated_local.values():
            _check_set_elements(s)


def check_data_requirements(study: Study) -> None:
    """Validate that the database supplies data for every parameter of every
    component defined in the system.

    Raises
    ------
    ValueError
        If a required data entry is missing or its time/scenario structure
        does not match what the model parameter expects.
    """
    for component in study.system.components:
        for param in component.model.parameters.values():
            data_structure = study.database.get_data(component.id, param.name)

            if not data_structure.check_requirement(
                component.model.parameters[param.name].structure.time,
                component.model.parameters[param.name].structure.scenario,
            ):
                raise ValueError(
                    f"Data inconsistency for component: {component.id}, "
                    f"parameter: {param.name}. Requirement not met."
                )
