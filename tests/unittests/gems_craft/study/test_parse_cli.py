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
"""Unit tests for gems_craft.study.parsing.parse_cli."""

import sys
from pathlib import Path

import pytest

from gems_craft.study.parsing import parse_cli


def test_output_format_defaults_to_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gemspy", "--study", "my_study"])
    parsed = parse_cli()
    assert parsed.study_dir == Path("my_study")
    assert parsed.output_format == "csv"


def test_output_format_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["gemspy", "--study", "my_study", "--output-format", "parquet"]
    )
    assert parse_cli().output_format == "parquet"


def test_output_format_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["gemspy", "--study", "my_study", "--output-format", "xlsx"]
    )
    with pytest.raises(SystemExit):
        parse_cli()
