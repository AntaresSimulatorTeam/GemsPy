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


from gems_craft.study.parsing import SystemSchema
from gems_craft.utils import ModifiedBaseModel


class AreaConnectionsSchema(ModifiedBaseModel):
    component: str
    port: str
    area: str


class ThermalComponentSchema(ModifiedBaseModel):
    area: str
    cluster_id: str


class ThermalCapacityConnectionSchema(ModifiedBaseModel):
    component: str
    port: str
    thermal_component: ThermalComponentSchema


class HybridSystemSchema(SystemSchema):
    area_connections: Optional[List[AreaConnectionsSchema]] = None
    thermal_capacity_connections: Optional[List[ThermalCapacityConnectionSchema]] = None
