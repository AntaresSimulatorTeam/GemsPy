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
Module for common classes used in models.
"""
from enum import Enum


class ValueType(Enum):
    CONTINUOUS = "CONTINUOUS"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"


class ProblemContext(Enum):
    OPERATIONAL = 0
    INVESTMENT = 1
    COUPLING = 2
