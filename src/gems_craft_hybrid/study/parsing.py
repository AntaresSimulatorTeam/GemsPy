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

from typing import List, Optional

from gems_craft.study.parsing import AreaConnectionsSchema, SystemSchema
from gems_craft.utils import ModifiedBaseModel

__all__ = [
    "AreaConnectionsSchema",
    "ThermalComponentSchema",
    "ThermalCapacityConnectionSchema",
    "HybridSystemSchema",
]


class ThermalComponentSchema(ModifiedBaseModel):
    area: str
    cluster_id: str


class ThermalCapacityConnectionSchema(ModifiedBaseModel):
    component: str
    port: str
    thermal_component: ThermalComponentSchema


class HybridSystemSchema(SystemSchema):
    thermal_capacity_connections: Optional[List[ThermalCapacityConnectionSchema]] = None
