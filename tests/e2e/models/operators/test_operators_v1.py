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

import math
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from gems.model.parsing import InputLibrary, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import OutputValues, build_problem
from gems.simulation.time_block import TimeBlock
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import build_data_base, build_network, resolve_system


@pytest.fixture
def data_dir(request) -> Path:
    return request.param


@pytest.fixture
def results_dir(data_dir: Path) -> Path:
    return data_dir / "output"


@pytest.fixture
def input_dir(data_dir: Path) -> Path:
    return data_dir / "input"


@pytest.fixture
def system_file(input_dir: Path) -> Path:
    return input_dir / "system.yml"


@pytest.fixture
def series_dir(input_dir: Path) -> Path:
    return input_dir / "data-series"


@pytest.fixture
def optim_result_file(results_dir: Path) -> Path:
    return results_dir / "simulation_table--20260323-1953.csv"


@pytest.fixture
def batch() -> int:
    return 1


@pytest.fixture
def relative_accuracy() -> float:
    return 1e-6


@pytest.fixture
def input_libraries(input_dir: Path) -> List[InputLibrary]:
    libs_dir = input_dir / "model-libraries"
    with open(libs_dir / "test_lib.yml") as lib_file:
        lib_new = parse_yaml_library(lib_file)
    return [lib_new]


@pytest.mark.parametrize(
    "data_dir",
    [
        Path(__file__).parent / "optest1",
        Path(__file__).parent / "optest2",
        Path(__file__).parent / "optest3",
    ],
    indirect=True,
)
def test_model_behaviour(
    system_file: Path,
    optim_result_file: str,
    batch: int,
    relative_accuracy: float,
    input_libraries: List[InputLibrary],
    results_dir: Path,
    series_dir: Path,
) -> None:
    scenarios = 1

    # Hardcoded timestep range
    first_timestep = 0
    last_timestep = 167
    timesteps = list(range(first_timestep, last_timestep + 1))

    with open(system_file) as compo_file:
        input_component = parse_yaml_components(compo_file)

    result_lib = resolve_library(input_libraries)
    components_input = resolve_system(input_component, result_lib)
    database = build_data_base(input_component, Path(series_dir))
    network = build_network(components_input)
    df_ref = pd.read_csv(results_dir / optim_result_file)
    expected_objective = df_ref[df_ref["output"] == "OBJECTIVE_VALUE"]["value"].iloc[0]
    ref_gen3 = (
        df_ref[
            (df_ref["component"] == "unique_prod3") & (df_ref["output"] == "generation")
        ]
        .sort_values("block_time_index")["value"]
        .tolist()
    )

    for _ in range(0, batch):
        problem = build_problem(
            network,
            database,
            TimeBlock(1, timesteps),
            scenarios,
        )
        status = problem.solver.Solve()
        assert status == problem.solver.OPTIMAL
        assert math.isclose(
            problem.solver.Objective().Value(),
            problem.solver.Objective().BestBound(),
            rel_tol=relative_accuracy,
        )
        assert math.isclose(
            expected_objective,
            problem.solver.Objective().Value(),
            rel_tol=relative_accuracy,
        )

        output = OutputValues(problem)
        gen3_values = output.component("unique_prod3").var("generation").value[0]

        for t, (ref_val, sol_val) in enumerate(zip(ref_gen3, gen3_values)):
            assert math.isclose(
                ref_val,
                sol_val,
                rel_tol=relative_accuracy,
            ), f"unique_prod3.generation mismatch at timestep {t}: expected {ref_val}, got {sol_val}"
