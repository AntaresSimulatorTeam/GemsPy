"""
This module provides functions to load simulation studies from disk.

A study is defined by a directory containing:
- `input/system.yml`: A file describing the system to be simulated.
- `input/model-libraries/`: A folder containing model library files in YAML format.
- `input/data-series/`: A folder containing data series files.
"""

from pathlib import Path

from gems_runner.model.model import Model
from gems_craft.model.parsing import load_yaml_library
from gems_runner.model.resolve_library import resolve_library
from gems_craft.study.parsing import load_yaml_system
from gems_runner.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems_craft.study.scenario_builder import ScenarioBuilder
from gems_runner.study.study import Study


def load_study(study_dir: Path) -> Study:
    """
    Loads a study from a given directory.

    This function reads the system definition, model libraries, and data series
    from the study directory, resolves them, and builds the simulation system
    and database.

    Args:
        study_dir: The path to the study directory.

    Returns:
        A Study container holding the resolved system and database.
    """
    system_file = study_dir / "input" / "system.yml"
    lib_folder = study_dir / "input" / "model-libraries"
    series_dir = study_dir / "input" / "data-series"

    input_libraries = []
    for lib_file in lib_folder.glob("*.yml"):
        input_libraries.append(load_yaml_library(lib_file))

    input_study = load_yaml_system(system_file)
    lib_dict = resolve_library(input_libraries)
    system = resolve_system(input_study, lib_dict)
    model_dict: dict[str, Model] = {}
    for library in lib_dict.values():
        model_dict |= library.models
    consistency_check(system, model_dict)

    scenario_builder_path = (
        study_dir / "input" / "data-series" / "modeler-scenariobuilder.dat"
    )
    scenario_builder = (
        ScenarioBuilder.load_dat(scenario_builder_path)
        if scenario_builder_path.exists()
        else ScenarioBuilder()
    )
    database = build_data_base(input_study, series_dir, scenario_builder)
    return Study(system=system, database=database, scenario_builder=scenario_builder)
