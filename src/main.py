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
Script to convert a PyPSA study, loaded from NetCDF file, to Gems format and run it with Antares-modeler (Gems interpreter inside Antares)

"""

import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd
from pypsa import Network

from andromede.input_converter.src.logger import Logger
from andromede.pypsa_converter.utils import transform_to_yaml
from utils import  convert_pypsa_network, export_parameters_yaml, load_pypsa_study, replace_lines_by_links



script_dir = Path(__file__).resolve().parent
exec_path = script_dir.parent.parent / Path ("rte-antares-9.3.0-rc2-installer-64bits/bin/antares-modeler")
antares_studies_dir = script_dir.parent.parent / Path("Antares_Studies/")
template = antares_studies_dir / Path("template/")
pypsa_files_path = script_dir.parent / Path("tests/pypsa_converter/pypsa_input_files/")


def pypsa_to_antares(pypsa_network: Network, study_dir: str) -> None:
    """
    Convert a PyPSA network to an Antares study structure and export the required files.

    This function copies a predefined Antares template study into the given `study_dir`,
    converts the provided PyPSA network into Antares input formats (system YAML and 
    data-series), and writes out the required configuration files for an Antares study.

    Args:
        pypsa_network (Network): 
            The PyPSA Network object to convert.
        study_dir (str): 
            Path to the destination directory where the Antares study will be created. 
            The directory will be created if it does not exist and may be overwritten.

    Returns:
        None
    """
    shutil.copytree(template, study_dir, dirs_exist_ok=True)
    systems_dir = study_dir / Path("input/")
    series_dir = systems_dir / Path("data-series/")
    # Convert the PyPSA network in a Gems system
    input_system_from_pypsa_converter = convert_pypsa_network(
        pypsa_network.copy(), systems_dir, series_dir, ".tsv"
    )
    # Save the Gems sytem to .yml file
    system_filename = "system.yml"
    input_system_from_pypsa_converter.id = "test"
    transform_to_yaml(
        model=input_system_from_pypsa_converter,
        output_path=systems_dir / Path(system_filename),
    )
    # Save the parameters.yml file
    export_parameters_yaml(len(pypsa_network.timesteps)-1, study_dir / Path ("parameters.yml"))

def running_antares(exec_path: str, study_dir: str) -> float:
    """
    Run Antares-modeler on the specified study directory and return the objective value.

    Args:
        exec_path (str): Path to the Antares-modeler executable.
        study_dir (str): Path to the Antares study directory.

    Returns:
        float: Objective value from the Antares output solution file.
    """
    # Build the command as a list for safety and portability
    command = [str(exec_path), str(study_dir)]

    # Run the command and wait for it to finish
    try:
        result = subprocess.run(
            command, 
            check=True,          # raise CalledProcessError on failure
            capture_output=True, # capture stdout/stderr if needed
            text=True            # decode bytes to string
        )
        # Log or inspect result.stdout / result.stderr here if desired
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Antares execution failed with code {e.returncode}: {e.stderr}"
        )

    # Read the objective value from the solution file
    solution_file = Path(study_dir) / "output" / "solution.csv"
    df = pd.read_csv(solution_file, header=None, sep=" ")

    # Return the objective value at row 0, column 1
    return df.iloc[0, 1]

def main(nc_file: str, scale_load: float) -> None:
    """
    Main function to convert a PyPSA study into an Antares study, and to perform two parallel runs: in Antares and in PyPSA
    """
    # Set up logger
    logger = Logger(__name__, "")  
    
    # Loading the PyPSA network
    logger.info("Loading PyPSA study...")
    pypsa_network = load_pypsa_study(pypsa_files_path / Path(nc_file), scale_load)
    logger.info(
        f"Loaded PyPSA network with {len(pypsa_network.buses)} buses and {len(pypsa_network.generators)} generators"
    )
    # Adapting the PyPSA network: Given the fact that Lines are not yet implemented in the PyPSA > Gems converter, we transform the Lines into Links
    logger.info(f"Replacing {len(pypsa_network.lines)} Lines by links")
    pypsa_network = replace_lines_by_links(pypsa_network)
    T = len(pypsa_network.snapshots)
    logger.info(f"Number of timesteps: {T}")

    # Converting to Antares-modeler study
    logger.info("Converting PyPSA network to Gems format...")
    pypsa_name = nc_file.split(".")[0]
    study_dir = antares_studies_dir / Path( pypsa_name + "/")
    pypsa_to_antares(pypsa_network, study_dir)

    # Running Antares-Modeler
    logger.info(f"Launching Antares-modeler")
    t0 = time.time()
    antares_obj = running_antares(exec_path,study_dir)
    logger.info(f"Antares-modeler run completed in {time.time() - t0:.2f} seconds")

    # Running PyPSA
    logger.info("Solving PyPSA network after line to link...")
    t0 = time.time()
    pypsa_network.optimize()
    pypsa_obj = pypsa_network.objective + pypsa_network.objective_constant
    logger.info(f"PyPSA run completed in {time.time() - t0:.2f} seconds")

    # Comparing objectives

    logger.info(f"PyPSA objective value: {pypsa_obj}")
    logger.info(f"Antares objective value: {antares_obj}")
    
    assert math.isclose(
        pypsa_obj,
        antares_obj,
        rel_tol=1e-6,
    ), "Objectives do not match within tolerance"


main("simple.nc",1.5)
main("ac-dc-data.nc",0.4)
main("ac-dc-data.nc",0.7)
#pypsa_to_antares_e2e("ac-dc-data.nc",0.5)
#pypsa_to_antares_e2e("base_s_4_elec.nc", 0.6)
#pypsa_to_antares_e2e("base_s_6_elec_lvopt_.nc", 0.3)
