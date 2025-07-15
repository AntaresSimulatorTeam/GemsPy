from pathlib import Path

from pypsa import Network

from andromede.input_converter.src.logger import Logger
from andromede.pypsa_converter.pypsa_converter import PyPSAStudyConverter
from andromede.simulation.optimization import OptimizationProblem, build_problem
from andromede.simulation.time_block import TimeBlock
from andromede.study.parsing import InputSystem
from andromede.study.resolve_components import System, build_data_base, build_network


def convert_pypsa_network(
    pypsa_network: Network, systems_dir: Path, series_dir: Path, series_file_format: str
) -> InputSystem:
    """
    Convert a PyPSA network to an Andromede InputSystem.

    Args:
        pypsa_network: The PyPSA network to convert
        systems_dir: Directory to store system files
        series_dir: Directory to store time series data

    Returns:
        InputSystem: The converted Andromede InputSystem
    """
    logger = Logger(__name__, "")
    converter = PyPSAStudyConverter(
        pypsa_network, logger, systems_dir, series_dir, series_file_format
    )
    input_system_from_pypsa_converter = converter.to_gems_study()
    return input_system_from_pypsa_converter


def build_problem_from_system(
    resolved_system: System, input_system: InputSystem, series_dir: Path, timesteps: int
) -> OptimizationProblem:
    """
    Build an optimization problem from a resolved system.

    Args:
        resolved_system: The resolved Andromede system
        input_system: The input system
        series_dir: Directory containing time series data
        timesteps: Number of timesteps in the simulation

    Returns:
        OptimizationProblem: The built optimization problem
    """
    database = build_data_base(input_system, Path(series_dir))
    network = build_network(resolved_system)
    problem = build_problem(
        network,
        database,
        TimeBlock(1, [i for i in range(timesteps)]),
        1,
    )
    return problem
