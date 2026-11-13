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
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set

from gems_craft.model import Model, PortType


@dataclass(frozen=True)
class Library:
    id: str
    port_types: Dict[str, PortType]
    models: Dict[str, Model]
    taxonomy: Optional[str] = None
    sets: Set[str] = field(default_factory=set)


def library(
    id: str,
    port_types: Iterable[PortType],
    models: Iterable[Model],
    taxonomy: Optional[str] = None,
    sets: Optional[Iterable[str]] = None,
) -> Library:
    return Library(
        id=id,
        port_types=dict((p.id, p) for p in port_types),
        models=dict((m.id, m) for m in models),
        taxonomy=taxonomy,
        sets=set(sets) if sets else set(),
    )
