from pathlib import Path

from pypsa import Network
import yaml

from gems.input_converter.src.logger import Logger
from gems.pypsa_converter.pypsa_converter import PyPSAStudyConverter
from gems.simulation.optimization import OptimizationProblem, build_problem
from gems.simulation.time_block import TimeBlock
from gems.study.parsing import InputSystem
from gems.study.resolve_components import System, build_data_base, build_network



def pypsa_to_gems(
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


def load_pypsa_study(file: str) -> Network:
    """
    Load a PyPSA study from a NetCDF file, preparing it for analysis or manipulation.

    This function loads a PyPSA network from a predefined NetCDF file located in the
    pypsa_input_files directory. It uses a relative path to avoid hardcoding the
    absolute path.

    Returns:
        pypsa.Network: A PyPSA network object loaded from the NetCDF file,
                      containing all components and settings from the dataset.
    """
    from pathlib import Path

    import pypsa

    # Get the directory of the current file
    current_dir = Path(__file__).parent

    # Define the relative path to the input file
    input_file = current_dir / "pypsa_input_files" / file

    # Load the PyPSA network from the file
    network = pypsa.Network(input_file)

    return network





def replace_lines_by_links(network: Network) -> Network:
    """
    Replace lines in a PyPSA network with equivalent links.

    This function converts transmission lines to links, which allows for more
    flexible modeling of power flow constraints. Each line is replaced with
    two links (one for each direction) to maintain bidirectional flow capability.

    Args:
        network (pypsa.Network): The PyPSA network to modify

    Returns:
        pypsa.Network: The modified network with lines replaced by links
    """

    # Create a copy of the lines DataFrame to iterate over
    lines = network.lines.copy()

    # For each line, create two links (one for each direction)
    for idx, line in lines.iterrows():
        # Get line parameters
        bus0 = line["bus0"]
        bus1 = line["bus1"]
        s_nom = line["s_nom"]
        efficiency = 1.0

        # Add forward link
        network.add(
            "Link",
            f"{idx}-link-{bus0}-{bus1}",
            bus0=bus0,
            bus1=bus1,
            p_min_pu=-1,
            p_max_pu=1,
            p_nom=s_nom,  # Use line capacity as link capacity
            efficiency=efficiency,
        )
    network.remove("Line", lines.index)
    return network




def export_parameters_file(last_time_step: int, filepath: str = "config.yaml"):
    """
    Export a YAML file with default parameters,
    and the provided `last-time-step` value.

    Args:
        last_time_step (int): Value of the last time step.
        filepath (str): Path to the YAML file to create.
    """
    config = {
        "solver": "highs",
        "solver-logs": True,
        "solver-parameters": "THREADS 1",
        "no-output": False,
        "first-time-step": 0,
        "last-time-step": last_time_step
    }

    # Write the YAML file
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)

