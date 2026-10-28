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

from pathlib import Path
from typing import Any, Dict

from yaml import safe_dump

from gems_craft.model.parsing import LibrarySchema


def write_yaml_hybrid_library(library: LibrarySchema, path: Path) -> None:
    """
    An area-connection still writes its three mandatory keys, as null when unused. 
    price is omitted when empty, because earlier hybrid files reject that key.
    """
    data = library.model_dump(by_alias=True, exclude_none=True, mode="json")
    for port_type, dumped in zip(library.port_types, data["port-types"]):
        connection = port_type.area_connection
        if connection is None:
            continue
        area_connection: Dict[str, Any] = {
            "injection-to-balance": connection.injection_to_balance,
            "spillage-bound": connection.spillage_bound,
            "unsupplied-energy-bound": connection.unsupplied_energy_bound,
        }
        if connection.price is not None:
            area_connection["price"] = connection.price
        dumped["area-connection"] = area_connection
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        safe_dump({"library": data}, f, allow_unicode=True, sort_keys=False)
