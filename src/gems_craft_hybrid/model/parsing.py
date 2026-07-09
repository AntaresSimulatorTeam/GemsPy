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

from pydantic import Field

from gems_craft.model.parsing import LibrarySchema, PortTypeSchema
from gems_craft.utils import ModifiedBaseModel


class AreaConnectionSchema(ModifiedBaseModel):
    injection_to_balance: Optional[str] = None
    spillage_bound: Optional[str] = None
    unsupplied_energy_bound: Optional[str] = None


class PortThermalCapacitySchema(ModifiedBaseModel):
    capacity_field: str


class HybridPortTypeSchema(PortTypeSchema):
    area_connection: Optional[AreaConnectionSchema] = None
    thermal_capacity_connection: Optional[PortThermalCapacitySchema] = None


class HybridLibrarySchema(LibrarySchema):
    port_types: List[HybridPortTypeSchema] = Field(default_factory=list)  # type: ignore[assignment]
