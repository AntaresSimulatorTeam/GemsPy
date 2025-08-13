
# Standard library imports
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import pandas as pd
import pytest

# Local application/library imports
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import OutputValues, TimeBlock, build_problem
from gems.simulation.simulation_table import SimulationTable
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    build_network,
    resolve_system,
)

@pytest.mark.parametrize("scenario_count", [1, 3])
def test_pypsa_model_simulation_table(tmp_path, scenario_count):
    # --- Paths ---
    pypsa_library_file = Path("src/gems/libs/pypsa_models/pypsa_models.yml")
    pypsa_model_file = Path("tests/pypsa_converter/systems/pypsa_study.yml")
    database_path = Path("tests/pypsa_converter/series")

    # --- Load PyPSA library ---
    with pypsa_library_file.open() as lib_file:
        input_library_obj = parse_yaml_library(lib_file)

    # Resolve the library
    resolved_lib = resolve_library([input_library_obj])

    # --- Load PyPSA model ---
    with pypsa_model_file.open() as model_file:
        input_system = parse_yaml_components(model_file)

    # Resolve system and database
    components_input = resolve_system(input_system, resolved_lib)
    database = build_data_base(input_system, database_path)
    network = build_network(components_input)

    # --- Build and solve optimization problem ---
    time_block = TimeBlock(1, list(range(5)))  # adjust time steps as needed
    problem = build_problem(network, database, time_block, 1)

    status = problem.solver.Solve()
    assert status == problem.solver.OPTIMAL, "Problem did not solve optimally"

    # --- Extract output values ---
    results = OutputValues(problem)

    # --- Fill simulation table ---
    simu_table = SimulationTable()
    block_size = len(time_block.timesteps)
    simu_table.fill_from_output_values(
        results,
        block=1,
        absolute_time_offset=0,
        block_size=block_size,
        scenario_count=scenario_count
    )

    # --- Write CSV to temporary path ---
    csv_path = simu_table.write_csv(tmp_path, optim_nb=1)

    # --- Assertions ---
    assert csv_path.exists(), "Simulation table CSV not created"
    assert not simu_table.df.empty, "Simulation table dataframe is empty"

    # Optional: print first few rows
    print(simu_table.df.head())

