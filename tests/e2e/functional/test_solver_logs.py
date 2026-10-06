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
Regression test: solver-options.logs controls HiGHS console output.

Solves the 13_1 study through SimulationSession and captures the output HiGHS
writes to the process file descriptors. HiGHS always prints its "Running HiGHS"
banner when the solver is created, before options apply; everything after it is
controlled by logs.
"""

from pathlib import Path

import pytest

from gems_craft.optim_config.parsing import SolverOptionsConfig, load_optim_config
from gems_craft.study.folder import load_study
from gems_runner.session.session import SimulationSession

_STUDY_DIR = Path(__file__).parent / "studies" / "13_1"


def _solve_and_capture(capfd: pytest.CaptureFixture[str], logs: bool) -> str:
    optim_config = load_optim_config(_STUDY_DIR / "input" / "optim-config.yml")
    assert optim_config is not None
    optim_config = optim_config.model_copy(
        update={"solver_options": SolverOptionsConfig(name="highs", logs=logs)}
    )
    capfd.readouterr()  # drop output from loading the study

    table = SimulationSession(load_study(_STUDY_DIR), optim_config).run()

    objective = table.data.loc[table.data["output"] == "objective-value", "value"]
    assert objective.iloc[0] == pytest.approx(91_000)
    captured = capfd.readouterr()
    return captured.out + captured.err


@pytest.mark.parametrize("logs", [True, False])
def test_no_unknown_solver_option(
    capfd: pytest.CaptureFixture[str], logs: bool
) -> None:
    output = _solve_and_capture(capfd, logs)
    assert "is unknown" not in output
    assert "solver_logs" not in output


def test_logs_true_prints_the_solver_log(capfd: pytest.CaptureFixture[str]) -> None:
    output = _solve_and_capture(capfd, logs=True)
    assert "Model status" in output


def test_logs_false_silences_the_solver_log(capfd: pytest.CaptureFixture[str]) -> None:
    output = _solve_and_capture(capfd, logs=False)
    assert "Model status" not in output
    assert "Presolving model" not in output
