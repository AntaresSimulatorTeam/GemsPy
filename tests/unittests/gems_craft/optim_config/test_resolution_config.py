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

import pytest
from pydantic import ValidationError

from gems_craft.optim_config.parsing import ResolutionConfig, ResolutionMode

# ---------------------------------------------------------------------------
# Defaults and parsing
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    cfg = ResolutionConfig()
    assert cfg.mode == ResolutionMode.FRONTAL
    assert cfg.block_length is None
    assert cfg.block_overlap == 0
    assert cfg.carry_over_length is None
    assert cfg.effective_carry_over_length == 0


def test_kebab_case_aliases() -> None:
    cfg = ResolutionConfig.model_validate(
        {
            "mode": "sequential-subproblems",
            "block-length": 168,
            "block-overlap": 24,
            "carry-over-length": 12,
        }
    )
    assert cfg.block_length == 168
    assert cfg.block_overlap == 24
    assert cfg.carry_over_length == 12


def test_block_length_required_for_windowed_modes() -> None:
    with pytest.raises(ValidationError, match="block_length"):
        ResolutionConfig(mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS)


# ---------------------------------------------------------------------------
# effective_carry_over_length resolution
# ---------------------------------------------------------------------------


def test_carry_over_defaults_to_block_overlap() -> None:
    cfg = ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=168,
        block_overlap=24,
    )
    assert cfg.carry_over_length is None
    assert cfg.effective_carry_over_length == 24


def test_explicit_zero_is_distinct_from_unset() -> None:
    cfg = ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=168,
        block_overlap=24,
        carry_over_length=0,
    )
    assert cfg.carry_over_length == 0
    assert cfg.effective_carry_over_length == 0


def test_partial_carry_over() -> None:
    cfg = ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=10,
        block_overlap=4,
        carry_over_length=3,
    )
    assert cfg.effective_carry_over_length == 3


# ---------------------------------------------------------------------------
# block-overlap validation
# ---------------------------------------------------------------------------


def test_negative_block_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="'block-overlap' must be >= 0"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=-1,
        )


def test_block_overlap_equal_to_block_length_rejected() -> None:
    with pytest.raises(ValidationError, match="must be < 'block-length'"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=10,
        )


def test_block_overlap_greater_than_block_length_rejected() -> None:
    with pytest.raises(ValidationError, match="must be < 'block-length'"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=11,
        )


# ---------------------------------------------------------------------------
# carry-over-length validation
# ---------------------------------------------------------------------------


def test_negative_carry_over_length_rejected() -> None:
    with pytest.raises(ValidationError, match="'carry-over-length' must be >= 0"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=2,
            carry_over_length=-1,
        )


def test_carry_over_length_greater_than_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="must be <= 'block-overlap'"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=2,
            carry_over_length=3,
        )


def test_carry_over_length_rejected_when_overlap_is_zero() -> None:
    # No special case at block_overlap == 0: any positive carry-over-length
    # is rejected, there is no implicit single-timestep seeding any more.
    with pytest.raises(ValidationError, match="must be <= 'block-overlap'"):
        ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            block_length=10,
            block_overlap=0,
            carry_over_length=1,
        )


def test_carry_over_length_equal_to_overlap_accepted() -> None:
    cfg = ResolutionConfig(
        mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
        block_length=10,
        block_overlap=4,
        carry_over_length=4,
    )
    assert cfg.effective_carry_over_length == 4
