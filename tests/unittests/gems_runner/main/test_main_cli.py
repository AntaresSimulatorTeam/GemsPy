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
"""Unit tests for gems_runner.main.main.main_cli (the gemspy entry point)."""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import gems_runner.main.main as main_module
from gems_craft.study.parsing import OutputFormat


def _run_main_cli(monkeypatch: pytest.MonkeyPatch, argv: List[str]) -> Dict[str, Any]:
    """Run main_cli with *argv* and return the arguments passed to run_study."""
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(main_module, "run_study", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["gemspy", *argv])
    main_module.main_cli()
    assert len(calls) == 1
    return calls[0]


def test_main_cli_passes_output_format_to_run_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _run_main_cli(
        monkeypatch, ["--study", "my_study", "--output-format", "parquet"]
    )
    assert kwargs == {
        "study_dir": Path("my_study"),
        "optim_config_path": None,
        "output_format": OutputFormat.PARQUET,
    }


def test_main_cli_defaults_to_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _run_main_cli(monkeypatch, ["--study", "my_study"])
    assert kwargs["output_format"] is OutputFormat.CSV
