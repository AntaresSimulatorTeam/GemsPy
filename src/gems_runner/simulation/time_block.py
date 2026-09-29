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
from typing import List, Optional


@dataclass(frozen=True)
class TimeBlock:
    """
    One block for otimization (week in current tool).

    timesteps: list of the different timesteps of the block (0, 1, ... 168 for each hour in one week)
    """

    id: int
    timesteps: List[int]


def compute_blocks(
    first_time_step: int,
    last_time_step: int,
    block_length: Optional[int],
    block_overlap: int = 0,
) -> List[TimeBlock]:
    """Partition ``[first_time_step, last_time_step]`` (inclusive) into TimeBlocks.

    ``block_length=None`` returns a single full-horizon TimeBlock (id ``0``).
    Otherwise blocks are ``block_length`` timesteps wide, stepped by
    ``block_length - block_overlap``; the final block is truncated to the
    horizon. ``block_overlap=0`` (the default) produces non-overlapping
    windows, as used by ``parallel-subproblems`` and ``benders-decomposition``.
    """
    if block_length is None:
        return [TimeBlock(0, list(range(first_time_step, last_time_step + 1)))]
    delta = block_length - block_overlap
    blocks: List[TimeBlock] = []
    t_start = first_time_step
    block_id = 0
    while t_start < last_time_step + 1:
        end = min(t_start + block_length, last_time_step + 1)
        blocks.append(TimeBlock(block_id, list(range(t_start, end))))
        t_start += delta
        block_id += 1
    return blocks
