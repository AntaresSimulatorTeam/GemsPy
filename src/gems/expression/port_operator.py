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

"""
Operators that allow port manipulation of expressions
"""

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class PortOperator(ABC):
    pass


@dataclass(frozen=True)
class PortAggregator:
    pass


@dataclass(frozen=True)
class PortSum(PortAggregator):
    pass
