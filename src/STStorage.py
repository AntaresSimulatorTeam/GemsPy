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

from andromede.model.parsing import InputLibrary, parse_yaml_library
from andromede.model.resolve_library import resolve_library
from andromede.simulation.optimization import build_problem
from andromede.simulation.time_block import TimeBlock
from andromede.study.parsing import parse_yaml_components
from andromede.study.resolve_components import (
    build_data_base,
    build_network,
    resolve_system,
)

# Parameters
timespan = 10
batch = 1
scenarios = 1

# Load shared libraries
with open("./STStorage/antares-historic-lib.yml") as lib_file:
    lib_historic = parse_yaml_library(lib_file)

with open("./STStorage/short_term_storage_no_shadowck_lib.yml") as lib_file:
    lib_ST = parse_yaml_library(lib_file)

with open("src/andromede/libs/reference_models/andromede_v1_models.yml") as lib_file:
    lib_v1 = parse_yaml_library(lib_file)

input_libraries = [lib_historic, lib_ST, lib_v1]

# Define a helper function to run the model in batch mode
def run_model_batched(component_path, label):
    with open(component_path) as compo_file:
        input_system = parse_yaml_components(compo_file)
    
    result_lib = resolve_library(input_libraries)
    components_input = resolve_system(input_system, result_lib)
    database = build_data_base(input_system, Path("./STStorage/series"))
    network = build_network(components_input)

    total_objective = 0
    for k in range(batch):
        problem = build_problem(
            network,
            database,
            TimeBlock(1, [i for i in range(k * timespan, (k + 1) * timespan)]),
            scenarios,
        )
        status = problem.solver.Solve()
        assert status == problem.solver.OPTIMAL
        obj_val = problem.solver.Objective().Value()
        print(f"{label} - Batch {k}: Objective Value = {obj_val}")
        total_objective += obj_val

    print(f"{label} - Total Objective Value over {batch} batches: {total_objective}")
    return total_objective

# Run both models
objective_no_shadowck = run_model_batched("./STStorage/bde_system_no_shadowck.yml", "No Shadowck")
objective_classic = run_model_batched("./STStorage/bde_system_classic.yml", "Classic (with Shadowck)")

# Comparison
difference = objective_no_shadowck - objective_classic
print(f"\nDifference (No Shadowck - Classic): {difference}")

