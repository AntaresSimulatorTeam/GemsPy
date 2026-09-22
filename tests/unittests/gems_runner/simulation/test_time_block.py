# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

"""Tests for `compute_blocks`, the shared time-horizon-to-TimeBlocks
partitioning helper used by `parallel-subproblems` and `benders-decomposition`
(week-block splitting)."""

from gems_runner.simulation.time_block import TimeBlock, compute_blocks


def test_block_length_none_returns_single_full_horizon_block() -> None:
    blocks = compute_blocks(0, 9, None)
    assert blocks == [TimeBlock(0, list(range(10)))]


def test_no_overlap_exact_division() -> None:
    blocks = compute_blocks(0, 9, 5)
    assert blocks == [
        TimeBlock(0, [0, 1, 2, 3, 4]),
        TimeBlock(1, [5, 6, 7, 8, 9]),
    ]


def test_no_overlap_truncated_final_block() -> None:
    blocks = compute_blocks(0, 11, 5)
    assert blocks == [
        TimeBlock(0, [0, 1, 2, 3, 4]),
        TimeBlock(1, [5, 6, 7, 8, 9]),
        TimeBlock(2, [10, 11]),
    ]


def test_non_zero_first_time_step() -> None:
    blocks = compute_blocks(3, 12, 5)
    assert blocks == [
        TimeBlock(0, [3, 4, 5, 6, 7]),
        TimeBlock(1, [8, 9, 10, 11, 12]),
    ]


def test_overlap_steps_by_block_length_minus_overlap() -> None:
    # block-length: 10, block-overlap: 4 -> delta = 6, same shape as the
    # worked example in docs/user-guide/optim-config.md's
    # sequential-subproblems section (there illustrated for a single
    # consecutive pair of blocks).
    blocks = compute_blocks(0, 17, 10, block_overlap=4)
    assert blocks == [
        TimeBlock(0, list(range(0, 10))),
        TimeBlock(1, list(range(6, 16))),
        TimeBlock(2, list(range(12, 18))),
    ]


def test_overlap_truncated_final_block() -> None:
    blocks = compute_blocks(0, 12, 5, block_overlap=2)
    # delta = 3: starts at 0, 3, 6, 9, 12
    assert blocks == [
        TimeBlock(0, [0, 1, 2, 3, 4]),
        TimeBlock(1, [3, 4, 5, 6, 7]),
        TimeBlock(2, [6, 7, 8, 9, 10]),
        TimeBlock(3, [9, 10, 11, 12]),
        TimeBlock(4, [12]),
    ]
