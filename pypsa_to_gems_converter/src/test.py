from pathlib import Path

from pypsa import Network
from pypsa_to_gems_converter.src.pypsa_converter import PyPSAStudyConverter
import logging
from pypsa_to_gems_converter.src.utils import transform_to_yaml
from gems.study.parsing import parse_yaml_components
from gems.model.parsing import parse_yaml_library

from gems.model.resolve_library import resolve_library
from gems.simulation.optimization import build_problem
from gems.simulation.time_block import TimeBlock
from gems.study.resolve_components import build_data_base, build_network, resolve_system


repo_root = Path("/home/rtei-beg-see-dp/Gems-Development/GemsPy")

# Pick the input NetCDF
data_dir = repo_root / "pypsa_to_gems_converter" / "src" / "tests" / "test_files"
network = Network(data_dir / "simple.nc")

# Choose where to dump the converted system and time series
systems_dir = repo_root / "pypsa_to_gems_converter" / "tmp" / "systems"
series_dir = repo_root / "pypsa_to_gems_converter" / "tmp" / "series"
systems_dir.mkdir(parents=True, exist_ok=True)
series_dir.mkdir(parents=True, exist_ok=True)


input_system_from_pypsa_converter = PyPSAStudyConverter(
    network, logging.Logger(__name__, ""), systems_dir, series_dir, ".csv"
).to_gems_study()

pypsa_study_as_yaml = transform_to_yaml(
    input_system_from_pypsa_converter, systems_dir / "pypsa_study.yml"
)

with open(systems_dir / "pypsa_study.yml") as compo_file:
    input_system = parse_yaml_components(compo_file)


lib_path = (
    repo_root / "pypsa_to_gems_converter" / "src" / "pypsa_models" / "pypsa_models.yml"
)


with open(lib_path) as lib_file:
    input_libraries = [parse_yaml_library(lib_file)]

# Resolve the PyPSA system with the PyPSA library
result_lib = resolve_library(input_libraries)
resolved_system = resolve_system(input_system, result_lib)

# Build the data base from the converted InputSystem
database = build_data_base(input_system, series_dir)

# Build a Gems network and optimisation problem, then solve it
gems_network = build_network(resolved_system)
timesteps = len(
    network.snapshots
)  # or len(network.timesteps) depending on PyPSA version

problem = build_problem(
    gems_network,
    database,
    TimeBlock(1, list(range(timesteps))),
    1,
)

status = problem.solver.Solve()
print("solver status:", status)
print("objective value:", problem.solver.Objective().Value())
