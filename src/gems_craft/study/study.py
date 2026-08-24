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


@dataclass
class Study:
    """
    Container that pairs a System (component topology and connections) with a
    DataBase (parameter values for those components).

    These two objects are always used together to build an optimisation
    problem.  ``Study`` gathers them into a single, coherent unit; the
    cross-validation of the pair lives in ``study/validation.py``
    (``check_data_requirements``).
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
