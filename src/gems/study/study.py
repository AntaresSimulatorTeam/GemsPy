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

from gems.model.model import Model
from gems.study.data import DataBase
from gems.study.scenario_builder import ScenarioBuilder
from gems.study.system import Component, System


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

                if not data_structure.check_requirement(
                    component.model.parameters[param.name].structure.time,
                    component.model.parameters[param.name].structure.scenario,
                ):
                    raise ValueError(
                        f"Data inconsistency for component: {component.id}, "
                        f"parameter: {param.name}. Requirement not met."
                    )
