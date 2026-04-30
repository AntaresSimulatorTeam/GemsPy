import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.study import Study
from gems.study.data import DataBase
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems.study.system import System


def setup_data(pypsa_dir: Path) -> Tuple[System, DataBase]:
    study_file = pypsa_dir / "input" / "system.yml"
    lib_file = pypsa_dir / "input" / "model-libraries" / "pypsa_models.yml"
    series_dir = pypsa_dir / "input" / "data-series"
    with lib_file.open() as lib:
        input_library = parse_yaml_library(lib)

    with study_file.open() as c:
        input_study = parse_yaml_components(c)
    lib_dict = resolve_library([input_library])
    system = resolve_system(input_study, lib_dict)
    consistency_check(system, lib_dict["pypsa_models"].models)

    database = build_data_base(input_study, series_dir)
    return system, database


def build_pypsa_problem(system: System, database: DataBase, time_horizon: int) -> float:
    scenarios = 1
    time_block = TimeBlock(1, list(range(time_horizon)))
    start = time.time()
    problem = build_problem(Study(system, database), time_block, list(range(scenarios)))
    end = time.time()
    print(f"Time elapsed for horizon {time_horizon}: {end - start:.4f}")
    return end - start


def run_pypsa_performance_scalability(pypsa_dir: Path) -> None:
    system, database = setup_data(pypsa_dir)
    durations = {}

    for horizon in np.linspace(1, 21, num=4):
        durations[int(horizon)] = build_pypsa_problem(system, database, int(horizon))

    duration_df = pd.DataFrame.from_dict(durations, orient="index")
    duration_df.columns = pd.Index(["build time"])
    print(duration_df)
    duration_df.to_csv(pypsa_dir / "pypsa_build_time_scalability.csv")


if __name__ == "__main__":
    pypsa_dir = Path(__file__).parent / "data_pypsa"
    run_pypsa_performance_scalability(pypsa_dir)
