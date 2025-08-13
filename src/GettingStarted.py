import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.logger import Logger
from gems.model.parsing import InputLibrary, parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.simulation.optimization import OptimizationProblem
from gems.simulation.output_values import OutputValues
from gems.simulation.simulation_table import SimulationTable
from gems.study.data import load_ts_from_txt
from gems.study.parsing import InputSystem, parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    build_network,
    consistency_check,
    resolve_system,
)




def to_mps_file(week_index, scenario_index):
    # Export the optimization problem as MPS file
    with open(f"problem_{week_index}_{scenario_index}.mps", "w") as f:
        f.write(problem.solver.ExportModelAsMpsFormat(False, False))
        f.close()



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
    TimeBlock(1, [i for i in range(0, 5)]),
    1,
)

status = problem.solver.Solve()
print(problem.solver.Objective().Value())

results = OutputValues(problem)






to_mps_file(1, 1)





# --- Test SimulationTable ---
absolute_time_offset = 0
block_size = len(range(5))  # in your TimeBlock above
scenario_count = 3

simu_table = SimulationTable()
simu_table.fill_from_output_values(results,
    block=1,
    absolute_time_offset=1,
    block_size=block_size,
    scenario_count=scenario_count
)


csv_path = simu_table.write_csv(Path("outputs"), optim_nb=1)

print(f"Simulation table saved at: {csv_path}")
print(simu_table.df.head())

