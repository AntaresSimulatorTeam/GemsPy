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

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

from gems_craft.model.model import Model
from gems_craft.study.data import DataBase
from gems_craft.study.scenario_builder import ScenarioBuilder
from gems_craft.study.system import Component, System


def _format_axes(time: bool, scenario: bool) -> str:
    axes = [name for name, on in (("time", time), ("scenario", scenario)) if on]
    return " and ".join(axes) if axes else "no axis"


@dataclass
class Study:
    """
    Container that pairs a System (component topology and connections) with a
    DataBase (parameter values for those components).

    These two objects are always used together to build an optimisation
    problem.  ``Study`` gathers them into a single, coherent unit and
    provides the cross-validation logic that was previously spread between
    ``DataBase.requirements_consistency`` and the callers of
    ``build_problem``.
    """

    system: System
    database: DataBase
    scenario_builder: ScenarioBuilder = field(default_factory=ScenarioBuilder)

    @cached_property
    def model_components(self) -> Dict[str, List[Component]]:
        """Components grouped by their model.id."""
        result: Dict[str, List[Component]] = defaultdict(list)
        for component in self.system.all_components:
            result[component.model.id].append(component)
        return dict(result)

    @cached_property
    def models(self) -> Dict[str, Model]:
        """All unique models in the system, keyed by model.id."""
        return {
            mk: components[0].model for mk, components in self.model_components.items()
        }

    # TODO: this is a second, disjoint consistency check alongside
    # resolve_components.consistency_check() — consider consolidating both
    # into one validation module for System/Study.
    def check_consistency(self) -> None:
        """Validate that the database supplies data for every parameter of every
        component defined in the system.

        Raises
        ------
        ValueError
            If a required data entry is missing or its time/scenario structure
            does not match what the model parameter expects.
        """
        for component in self.system.components:
            for param in component.model.parameters.values():
                data_structure = self.database.get_data(component.id, param.name)
                declared = param.structure

                if not data_structure.check_requirement(
                    declared.time, declared.scenario
                ):
                    actual = data_structure.structure()
                    raise ValueError(
                        f"Data inconsistency for component: {component.id}, "
                        f"parameter: {param.name}. The data varies along "
                        f"[{_format_axes(actual.time, actual.scenario)}] but the model "
                        f"{component.model.id!r} declares the parameter "
                        f"[{_format_axes(declared.time, declared.scenario)}]. Data may "
                        f"vary along fewer axes than the model declares, never more."
                    )
