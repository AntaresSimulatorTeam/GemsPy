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

_RUN_FOLDER = Path("my_study") / "output" / "20261008T1012-2"


def _run_main_cli(monkeypatch: pytest.MonkeyPatch, argv: List[str]) -> Dict[str, Any]:
    """Run main_cli with *argv* and return the arguments passed to run_study."""
    calls: List[Dict[str, Any]] = []

    def run_study(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return _RUN_FOLDER

    monkeypatch.setattr(main_module, "run_study", run_study)
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


def test_main_cli_prints_the_run_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run folder may have a -2, -3, ... suffix: the user is told where the
    results are."""
    _run_main_cli(monkeypatch, ["--study", "my_study"])
    assert capsys.readouterr().out == f"Results written to {_RUN_FOLDER}\n"
