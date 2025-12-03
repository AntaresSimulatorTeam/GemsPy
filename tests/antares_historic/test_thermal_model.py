import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pytest
from antares.craft import ThermalClusterGroup, ThermalClusterProperties

from gems.input_converter.src.logger import Logger
from tests.antares_historic.utils import (
    convert_study,
    createThermalTestAntaresStudy,
    first_optim_relgap,
)

LOAD_FILES_DIR = Path("tests/antares_historic/data")
THERMAL_TEST_REL_ACCURACY = 5 * 1e-5
THERMAL_TEST_SOLVER = "highs"


def thermal_test_procedure(
    study_name: str,
    study_path: Path,
    marg_cluster_properties: ThermalClusterProperties,
    load_time_serie_file: Path,
    exec_folder: Path,
) -> None:
    cluster_data_frame = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name,
        study_path,
        load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame,
    )
    original_study_path, converted_study_path = convert_study(
        study_path, study_name, ["thermal"]
    )
    rel_gap = first_optim_relgap(
        exec_folder, original_study_path, converted_study_path, THERMAL_TEST_SOLVER
    )
    assert rel_gap < THERMAL_TEST_REL_ACCURACY


@pytest.fixture(scope="session")
def cluster_list_general_test() -> list[ThermalClusterProperties]:

    return [
        ThermalClusterProperties(
            unit_count=2,
            nominal_capacity=100,
            marginal_cost=10,
            market_bid_cost=10,
            startup_cost=1000,
            fixed_cost=1000,
            min_down_time=2,
            min_up_time=2,
            group=ThermalClusterGroup.NUCLEAR,
        ),
        ThermalClusterProperties(
            unit_count=2,
            nominal_capacity=100,
            marginal_cost=27.7,
            market_bid_cost=27.7,
            startup_cost=10000,
            fixed_cost=100,
            min_down_time=2,
            min_up_time=2,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    ]  # ,
    """           ThermalClusterProperties(unit_count=2,
                                                                    nominal_capacity=100,
                                                                    marginal_cost=27.7,
                                                                    market_bid_cost=27.7,
                                                                    startup_cost= 100,
                                                                    fixed_cost= 100,
                                                                    min_down_time = 2,
                                                                    min_up_time = 4,
                                                                    group=ThermalClusterGroup.NUCLEAR),
                ThermalClusterProperties(unit_count=4,
                                                                    nominal_capacity=100,
                                                                    marginal_cost=127.7,
                                                                    market_bid_cost=127.7,
                                                                    startup_cost= 100,
                                                                    fixed_cost= 1000,
                                                                    min_down_time = 2,
                                                                    min_up_time = 4,
                                                                    group=ThermalClusterGroup.NUCLEAR),
                 ThermalClusterProperties(unit_count=4,
                                                                    nominal_capacity=100,
                                                                    marginal_cost=127.7,
                                                                    market_bid_cost=127.7,
                                                                    startup_cost= 100,
                                                                    fixed_cost= 1000,
                                                                    min_down_time = 2,
                                                                    min_up_time = 4,
                                                                    group=ThermalClusterGroup.NUCLEAR)]"""


def test_general_thermal(
    cluster_list_general_test: list[ThermalClusterProperties],
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    for load_time_serie_file in os.listdir(LOAD_FILES_DIR):
        for marg_cluster_properties in cluster_list_general_test:
            study_name = f"e2e_test_{str(time())}"
            thermal_test_procedure(
                study_name,
                auto_generated_studies_path,
                marg_cluster_properties,
                LOAD_FILES_DIR / load_time_serie_file,
                antares_exec_folder,
            )
