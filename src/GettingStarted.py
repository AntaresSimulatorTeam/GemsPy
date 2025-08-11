import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from andromede.input_converter.src.converter import AntaresStudyConverter
from andromede.input_converter.src.logger import Logger
from andromede.model.parsing import InputLibrary, parse_yaml_library
from andromede.model.resolve_library import resolve_library
from andromede.simulation import TimeBlock, build_problem
from andromede.simulation.optimization import OptimizationProblem
from andromede.study.data import load_ts_from_txt
from andromede.study.parsing import InputSystem, parse_yaml_components
from andromede.study.resolve_components import (
    build_data_base,
    build_network,
    consistency_check,
    resolve_system,
)

print("Working directory:", os.getcwd())
    
with open("./GettingStarted/system.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)

with open("./GettingStarted/library.yml") as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

result_lib = resolve_library(input_libraries)
components_input = resolve_system(input_system, result_lib)
database = build_data_base(input_system, Path("./GettingStarted"))

network = build_network(components_input)

problem = build_problem(
    network,
    database,
    TimeBlock(1, [i for i in range(0, 3)]),
    1,
)

status = problem.solver.Solve()
print(problem.solver.Objective().Value())