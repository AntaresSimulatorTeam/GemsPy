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

from gems_craft.optim_config.parsing import OptimConfig


@dataclass(frozen=True)
class TimeBlock:
    """
    One block for otimization (week in current tool).

    timesteps: list of the different timesteps of the block (0, 1, ... 168 for each hour in one week)
    """

    id: int
    timesteps: List[int]


def time_scope_block(optim_config: OptimConfig, block_id: int) -> TimeBlock:
    """One block covering the whole time scope (frontal and Benders modes)."""
    return TimeBlock(
        block_id,
        list(
            range(
                optim_config.time_scope.first_time_step,
                optim_config.time_scope.last_time_step + 1,
            )
        ),
    )


def parallel_blocks(optim_config: OptimConfig) -> List[TimeBlock]:
    """Split the time scope into consecutive blocks of ``block-length`` time
    steps, the last one possibly shorter (parallel-subproblems mode)."""
    block_length: int = optim_config.resolution.block_length  # type: ignore[assignment]
    first = optim_config.time_scope.first_time_step
    last = optim_config.time_scope.last_time_step
    return [
        TimeBlock(i, list(range(t, min(t + block_length, last + 1))))
        for i, t in enumerate(range(first, last + 1, block_length))
    ]
