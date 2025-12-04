import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from antares.craft import *

from antares_runner.antares_runner import AntaresHybridStudyBenchmarker
from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.data_preprocessing.data_classes import ConversionMode
from gems.input_converter.src.logger import Logger

ANTARES_VERSION_CREATED_STUDIES = "9.2"
ANTARES_LEGACY_MODELS_PATH = [
    Path(
        "tests/antares_historic/antares-resources/reference_libraries/antares_legacy_models.yml"
    )
]
ACCURATE_TEMPLATE_PATH = Path(
    "tests/antares_historic/antares-resources/hybrid_mode_addon/uc_accurate"
)


def convert_study(
    study_dir: Path, study_name: str, model_list: list[str]
) -> tuple[Path, Path]:
    """Take the study study_dir / study_name and generate a hybrid study where all the components of the listed models are converted in GEMS format, according library ANTARES_LEGACY_MODELS_PATH."""
    log_path = ""
    logger: logging.Logger = Logger(__name__, log_path)
    study_path = study_dir / study_name
    converter_output_folder = study_dir.parent / "antares-studies-converted/"
    params = {
        "study_input": study_path,
        "logger": logger,
        "mode": ConversionMode.HYBRID.value,
        "output_folder": converter_output_folder,
        "lib_paths": ANTARES_LEGACY_MODELS_PATH,
        "models_to_convert": model_list,
    }
    converter = AntaresStudyConverter(**params)  # type: ignore
    converter.process_all()
    return study_path, converter.output_folder


def first_optim_relgap(
    exec_folder: Path,
    study_path_1: Path,
    study_path_2: Path,
    solver: Optional[str] = None,
) -> float:
    benchmarker = AntaresHybridStudyBenchmarker(
        exec_folder, study_path_1, study_path_2, solver
    )
    benchmarker.run()
    rel_gaps = benchmarker.weekly_rel_gaps()
    return rel_gaps[0].max()


def addHybridBehavior(study_path: Path) -> None:
    """Function to add some files to a Legacy Antares Study, so as to generate an hybrid behaviour (generation of the simulation table) with no impact on the simulation."""
    shutil.copytree(
        ACCURATE_TEMPLATE_PATH / "input", study_path / "input", dirs_exist_ok=True
    )
    shutil.copy2(
        ACCURATE_TEMPLATE_PATH / "generaldata.ini",
        study_path / "settings" / "generaldata.ini",
    )


def createThermalTestAntaresStudy(
    study_name: str,
    parent_dir_path: Path,
    load_time_serie_file: Path,
    marg_cluster_properties: ThermalClusterProperties,
    marg_cluster_data_frame: pd.DataFrame,
) -> None:
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir_path,
    )
    load_timeserie = pd.read_csv(load_time_serie_file)
    area = study.create_area(
        area_name="unique", properties=AreaProperties(energy_cost_unsupplied=20000)
    )
    area.set_load(load_timeserie)
    cluster1 = area.create_thermal_cluster(
        "prod",
        ThermalClusterProperties(
            unit_count=2,
            nominal_capacity=150,
            marginal_cost=10,
            market_bid_cost=10,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    )
    cluster1.set_series(pd.DataFrame(data=150 * np.ones((8760, 1))))

    cluster2 = area.create_thermal_cluster(
        "prod2",
        ThermalClusterProperties(
            unit_count=1,
            nominal_capacity=200,
            marginal_cost=20,
            market_bid_cost=20,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    )
    cluster2.set_series(pd.DataFrame(data=200 * np.ones((8760, 1))))

    cluster3 = area.create_thermal_cluster("prod3", marg_cluster_properties)
    cluster3.set_series(marg_cluster_data_frame)
    addHybridBehavior(parent_dir_path / study_name)
