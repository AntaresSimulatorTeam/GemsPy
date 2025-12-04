# Copyright (c) 2025, RTE (https://www.rte-france.com)
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


import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


class AntaresHybridRunner:
    def __init__(
        self, exec_dir: Path, study_dir: Path, solver: None | str = None
    ) -> None:
        self.exec_dir, self.study_dir, self.solver = exec_dir, study_dir, solver
        self.exec_path = self.exec_dir / Path("antares-solver")
        self.OUTPUT_FILE_DIR_NAME = "output"
        self.SIMULATION_TABLE_1_FILE = "simulation_table--optim-nb-1.csv"
        self.SIMULATION_TABLE_2_FILE = "simulation_table--optim-nb-2.csv"

    def run(self) -> None:
        # Build the command to run antares-solver.exe
        command = [str(self.exec_path.absolute()), str(self.study_dir.absolute())]
        if self.solver:
            command.append("--linear-solver")
            command.append(self.solver)
        # Run the command and wait for it to finish
        try:
            start = time.time()
            subprocess.run(
                command,
                check=True,  # raise CalledProcessError on failure
                capture_output=True,  # capture stdout/stderr if needed
                text=True,  # decode bytes to string
            )
            # Log or inspect result.stdout / result.stderr here if desired
            end = time.time()
            self.total_execution_time = end - start
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Antares execution failed with code {e.returncode}: {e.stderr}"
            )
        # Read the objective value from the solution file
        path = self.study_dir / Path(self.OUTPUT_FILE_DIR_NAME)
        folders = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        self.simulation_id = sorted(folders)[-1]
        simulation_table_1_path, simulation_table_2_path = path / Path(
            self.simulation_id + "/" + self.SIMULATION_TABLE_1_FILE
        ), path / Path(self.simulation_id + "/" + self.SIMULATION_TABLE_2_FILE)
        self.simulation_table_1, self.simulation_table_2 = pd.read_csv(
            simulation_table_1_path
        ), pd.read_csv(simulation_table_2_path)


class AntaresHybridStudyBenchmarker:
    def __init__(
        self,
        exec_path: Path,
        study_path_1: Path,
        study_path_2: Path,
        solver: None | str = None,
    ) -> None:
        self.exec_path, self.study_path_1, self.study_path_2, self.solver = (
            exec_path,
            study_path_1,
            study_path_2,
            solver,
        )
        self.weekly_objectives_1: list = []
        self.weekly_objectives_2: list = []
        self.simulation_tables_1: list = []
        self.simulation_tables_2: list = []

    def run(self) -> None:
        for study in [self.study_path_1, self.study_path_2]:
            r = AntaresHybridRunner(self.exec_path, study, self.solver)
            r.run()
            self.weekly_objectives_1.append(
                r.simulation_table_1.query('output == "OBJECTIVE_VALUE"')[
                    "value"
                ].values
            )
            self.weekly_objectives_2.append(
                r.simulation_table_2.query('output == "OBJECTIVE_VALUE"')[
                    "value"
                ].values
            )
            self.simulation_tables_1.append(r.simulation_table_1)
            self.simulation_tables_2.append(r.simulation_table_2)

    def weekly_rel_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.abs(
            (self.weekly_objectives_1[0] - self.weekly_objectives_1[1])
            / self.weekly_objectives_1[0]
        ), np.abs(
            (self.weekly_objectives_2[0] - self.weekly_objectives_2[1])
            / self.weekly_objectives_2[0]
        )

    def weekly_abs_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.abs(
            self.weekly_objectives_1[0] - self.weekly_objectives_1[1]
        ), np.abs(self.weekly_objectives_2[0] - self.weekly_objectives_2[1])


class AntaresModelerRunner:
    def __init__(self, exec_dir: str, study_dir: str) -> None:
        self.exec_dir, self.study_dir = exec_dir, study_dir
        self.exec_path = self.exec_dir / Path("antares-modeler")
        self.OUTPUT_FILE_DIR_NAME = "output"

    def run(self) -> None:
        # Build the command to run antares-modeler.exe
        command = [str(self.exec_path), str(self.study_dir)]
        # Run the command and wait for it to finish
        try:
            start = time.time()
            subprocess.run(
                command,
                check=True,  # raise CalledProcessError on failure
                capture_output=True,  # capture stdout/stderr if needed
                text=True,  # decode bytes to string
            )
            # Log or inspect result.stdout / result.stderr here if desired
            end = time.time()
            self.total_execution_time = end - start
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Antares execution failed with code {e.returncode}: {e.stderr}"
            )
        # Read the objective value from the solution file
        path = self.study_dir / Path(self.OUTPUT_FILE_DIR_NAME)
        folders = [name for name in os.listdir(path)]
        self.simulation_file = sorted(folders)[-1]
        assert "simulation_table" in self.simulation_file
        simulation_table_path = path / Path(self.simulation_file)
        self.simulation_table = pd.read_csv(simulation_table_path)
