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

"""
Script to convert a PyPSA study, loaded from NetCDF file, to Gems format and run it.

This script loads a PyPSA study using the load_pypsa_study function,
converts it to Gems format, and runs the converted study.
"""

import math
import os
from pathlib import Path

from pypsa import Network

from andromede.input_converter.src.logger import Logger
from andromede.model.parsing import parse_yaml_library
from andromede.model.resolve_library import resolve_library
from andromede.pypsa_converter.utils import transform_to_yaml
from andromede.study.resolve_components import resolve_system
from tests.pypsa_converter.utils import build_problem_from_system, convert_pypsa_network


def load_pypsa_study(file: str, load_scaling: float) -> Network:
    """
    Load a PyPSA study from a NetCDF file, preparing it for analysis or manipulation.

    This function loads a PyPSA network from a predefined NetCDF file located in the
    pypsa_input_files directory. It uses a relative path to avoid hardcoding the
    absolute path.

    Returns:
        pypsa.Network: A PyPSA network object loaded from the NetCDF file,
                      containing all components and settings from the dataset.
    """
    import os
    from pathlib import Path

    import pypsa

    # Get the directory of the current file
    current_dir = Path(__file__).parent

    # Define the relative path to the input file
    input_file = current_dir / "pypsa_input_files" / file

    # Load the PyPSA network from the file
    network = pypsa.Network(input_file)

    # Scale the load to make the test case feasible
    network = scale_load(network, load_scaling)

    return network


def extend_quota(network: Network) -> Network:
    # Temporary function, used while the GlobalConstraint model is not implemented yet.
    # Set the CO2 bound to very large value
    network.global_constraints["constant"][0] = 10000000000
    return network


def scale_load(network: Network, factor: float) -> Network:
    network.loads_t["p_set"] *= factor
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


def pypsa_gemspy_benchmark(
    file: str, load_scaling: float, activate_quota: bool
) -> None:
    """
    Main function to convert a PyPSA study to Gems format and run it.
    """
    # Set up logger
    logger = Logger(__name__, "")

    # Define directories for systems and series
    current_dir = Path(__file__).parent
    systems_dir = current_dir / "systems"
    series_dir = current_dir / "series"

    # Create directories if they don't exist
    systems_dir.mkdir(exist_ok=True)
    series_dir.mkdir(exist_ok=True)

    # Load the PyPSA study
    logger.info("Loading PyPSA study...")
    pypsa_network = load_pypsa_study(file, load_scaling)
    logger.info(
        f"Loaded PyPSA network with {len(pypsa_network.buses)} buses and {len(pypsa_network.generators)} generators"
    )
    logger.info(f"Replacing {len(pypsa_network.lines)} Lines by links")
    pypsa_network = replace_lines_by_links(pypsa_network)
    if not (activate_quota):
        pypsa_network = extend_quota(pypsa_network)

    # Get the number of timesteps
    T = len(pypsa_network.snapshots)
    logger.info(f"Number of timesteps: {T}")
    # Convert to Gems System
    logger.info("Converting PyPSA network to Gems format...")
    input_system_from_pypsa_converter = convert_pypsa_network(
        pypsa_network.copy(), systems_dir, series_dir, ".txt"
    )

    # Save the InputSystem to YAML
    system_filename = "pypsa_study.yml"
    logger.info(f"Saving Gems system to {systems_dir / system_filename}...")
    transform_to_yaml(
        model=input_system_from_pypsa_converter,
        output_path=systems_dir / system_filename,
    )

    # Load the model library
    logger.info("Loading model library...")
    # Get the path to the project root by going up two levels from the current directory
    project_root = Path(__file__).parents[2]
    pypsa_models_path = (
        project_root / "src/andromede/libs/pypsa_models/pypsa_models.yml"
    )
    logger.info(f"Loading PyPSA models from {pypsa_models_path}...")
    with open(pypsa_models_path) as lib_file:
        input_libraries = [parse_yaml_library(lib_file)]
    result_lib = resolve_library(input_libraries)

    # Resolve the system
    logger.info("Resolving the system...")
    resolved_system = resolve_system(input_system_from_pypsa_converter, result_lib)

    # Build and solve the optimization problem
    logger.info("Building the optimization problem...")
    problem = build_problem_from_system(
        resolved_system, input_system_from_pypsa_converter, series_dir, T
    )

    logger.info("Solving the optimization problem...")
    # Solve the problem
    problem.solver.EnableOutput()
    status = problem.solver.Solve()

    # Log the results
    if status == problem.solver.OPTIMAL:
        logger.info("Optimization problem solved successfully!")
        logger.info(f"Objective value: {problem.solver.Objective().Value()}")
    else:
        logger.error(f"Failed to solve optimization problem. Status: {status}")

    # Optimize PyPSA network
    logger.info("Solving PyPSA network after line to link...")
    pypsa_network.optimize()
    logger.info(f"PyPSA objective value: {pypsa_network.objective}")
    assert math.isclose(
        pypsa_network.objective + pypsa_network.objective_constant,
        problem.solver.Objective().Value(),
        rel_tol=1e-6,
    )


def test_case_gemspy() -> None:
    pypsa_gemspy_benchmark("base_s_6_elec_lvopt_.nc", 0.4, True)
    pypsa_gemspy_benchmark("base_s_6_elec_lvopt_.nc", 0.3, True)
    pypsa_gemspy_benchmark("simple.nc", 1.0, False)
