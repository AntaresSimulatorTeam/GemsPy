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
from dataclasses import dataclass
from typing import Dict, Iterable

from gems.model import Model, PortType


@dataclass(frozen=True)
class Library:
    id: str
    port_types: Dict[str, PortType]
    models: Dict[str, Model]


def library(
    id: str,
    port_types: Iterable[PortType],
    models: Iterable[Model],
) -> Library:
    return Library(
        id=id,
        port_types=dict((p.id, p) for p in port_types),
        models=dict((m.id, m) for m in models),
    )
