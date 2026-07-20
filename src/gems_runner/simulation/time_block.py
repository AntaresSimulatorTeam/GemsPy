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
from typing import List


@dataclass(frozen=True)
class TimeBlock:
    """
    One block for otimization (week in current tool).

    timesteps: list of the different timesteps of the block (0, 1, ... 168 for each hour in one week)
    """

    id: int
    timesteps: List[int]
