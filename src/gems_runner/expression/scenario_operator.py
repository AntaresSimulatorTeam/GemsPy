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
Operators that allow manipulation of expressions with respect to scenarios
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioOperator(ABC):
    def __str__(self) -> str:
        return NotImplemented

    @classmethod
    @abstractmethod
    def degree(cls) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Expectation(ScenarioOperator):
    def __str__(self) -> str:
        return "expec()"

    @classmethod
    def degree(cls) -> int:
        return 1


@dataclass(frozen=True)
class Variance(ScenarioOperator):
    def __str__(self) -> str:
        return "variance()"

    @classmethod
    def degree(cls) -> int:
        return 2
