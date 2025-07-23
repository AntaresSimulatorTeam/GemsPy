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
Describes probability distributions used in the models
"""

from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from gems.expression.expression import ExpressionNode


class AbstractProbabilityLaw(ABC):
    def get_sample(self, size: int) -> List[float]:
        return NotImplemented


@dataclass(frozen=True)
class Normal(AbstractProbabilityLaw):
    mean: ExpressionNode
    standard_deviation: ExpressionNode

    def get_sample(self, size: int) -> List[float]:
        return NotImplemented


@dataclass(frozen=True)
class Uniform(AbstractProbabilityLaw):
    lower_bound: ExpressionNode
    upper_bound: ExpressionNode

    def get_sample(self, size: int) -> List[float]:
        return NotImplemented


@dataclass(frozen=True)
class UniformIntegers(AbstractProbabilityLaw):
    lower_bound: ExpressionNode
    upper_bound: ExpressionNode

    def get_sample(self, size: int) -> List[float]:
        return NotImplemented
