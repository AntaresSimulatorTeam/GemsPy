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

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "THERMAL CLUSTER"  : CONSTANT DATA  ##
LOAD_FILES_DIR = Path("tests/antares_historic/data")
THERMAL_TEST_REL_ACCURACY = 5 * 1e-5
THERMAL_TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2
LOAD_TIME_SERIE_FILES = [
    "load_matrix_1.txt",
    # "load_matrix_2.txt", #uncomment to test with different load profile
    # "load_matrix_original.txt",
]

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "THERMAL CLUSTERS"  : TESTS  ##

# General tests [OK:test_general_thermal]

# Testing Boolean/discrete parameters
## group: [TODO]
## gen_ts: [TODO]
## must_run: [TODO]
## law_forced: [TODO]
## law_planned: [TODO]
## cost_generation: [TODO]


# Testing Float parameters
## nominal_capacity [OK : test_nominal_capacity]
## min_stable_power: [OK : test_min_stable_power]
## min_up_time: [OK : test_min_up_time]
## min_down_time: [OK : test_min_down_time]
## marginal_cost: [OK : test_marginal_cost_marketbid_equals]
## spread_cost: [TODO]
## fixed_cost: [OK : test_fixed_cost]
## startup_cost: [OK : test_startup_cost]
## market_bid_cost: [OK : test_marginal_cost_marketbid_equals]
## spinning: [TODO]
## volatility_forced: [TODO]
## volatility_planned: [TODO]
## efficiency:  [TODO]
## variable_o_m_cost: [TODO]
## co2, nh3, so2:, nox, pm2_5, pm5, pm10, nmvoc, op1, op2, op3, op4, op5 [TODO]


# Testing Timeseries parameters
## series (availability) : [TODO]
## co2_cost_matrix : [TODO]
## fuel_cost_matrix : [TODO]


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
            nominal_capacity=100,
            marginal_cost=10,
            market_bid_cost=10,
            fixed_cost=1000,
            group=ThermalClusterGroup.NUCLEAR,
        ),
        ThermalClusterProperties(
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
            nominal_capacity=100,
            marginal_cost=27.7,
            market_bid_cost=27.7,
            startup_cost=10000,
            fixed_cost=100,
            min_down_time=2,
            min_up_time=2,
            group=ThermalClusterGroup.NUCLEAR,
        ),
        ThermalClusterProperties(
            nominal_capacity=100,
            marginal_cost=200,
            market_bid_cost=200,
            startup_cost=10000,
            fixed_cost=100,
            min_down_time=2,
            min_up_time=5,
            group=ThermalClusterGroup.NUCLEAR,
        ),
    ]  # ,


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_general_thermal(
    cluster_list_general_test: list[ThermalClusterProperties],
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    for marg_cluster_properties in cluster_list_general_test:
        study_name = f"e2e_general_test_{str(int(100*time()))}"
        thermal_test_procedure(
            study_name,
            auto_generated_studies_path,
            marg_cluster_properties,
            LOAD_FILES_DIR / load_time_serie_file,
            antares_exec_folder,
        )


@pytest.mark.parametrize("base_capacity", [50, 100, 200])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_nominal_capacity(
    base_capacity: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Setup thermal cluster properties
    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=base_capacity,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=500,
        group=ThermalClusterGroup.NUCLEAR,
    )

    # Run base test
    study_name_base = f"e2e_capacity_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY

    # Test +20% capacity
    marg_cluster_plus = ThermalClusterProperties(
        nominal_capacity=base_capacity * MODIFICATION_RATIO,
        marginal_cost=10,
        market_bid_cost=10,
        fixed_cost=500,
        group=ThermalClusterGroup.NUCLEAR,
    )
    study_name_plus = f"e2e_capacity_plus_{str(int(100*time()))}"
    cluster_data_frame_plus = pd.DataFrame(
        data=marg_cluster_plus.unit_count
        * marg_cluster_plus.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_plus,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_plus,
        cluster_data_frame_plus,
    )
    orig_path_plus = auto_generated_studies_path / study_name_plus
    rel_gap_plus = first_optim_relgap(
        antares_exec_folder, orig_path_plus, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_plus > THERMAL_TEST_REL_ACCURACY

    # Test -20% capacity
    marg_cluster_minus = ThermalClusterProperties(
        nominal_capacity=base_capacity / MODIFICATION_RATIO,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=500,
        group=ThermalClusterGroup.NUCLEAR,
    )
    study_name_minus = f"e2e_capacity_minus_{str(int(100*time()))}"
    cluster_data_frame_minus = pd.DataFrame(
        data=marg_cluster_minus.unit_count
        * marg_cluster_minus.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_minus,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_minus,
        cluster_data_frame_minus,
    )
    orig_path_minus = auto_generated_studies_path / study_name_minus
    rel_gap_minus = first_optim_relgap(
        antares_exec_folder, orig_path_minus, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_minus > THERMAL_TEST_REL_ACCURACY


@pytest.mark.parametrize("base_startup_cost", [50, 1000, 5000])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_startup_cost(
    base_startup_cost: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Setup thermal cluster properties with base startup cost
    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=100,
        marginal_cost=20,
        market_bid_cost=20,
        startup_cost=base_startup_cost,
        fixed_cost=500,
        group=ThermalClusterGroup.NUCLEAR,
    )

    # Run base test
    study_name_base = f"e2e_startup_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        # Test +/-20% startup cost
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=marg_cluster_properties.marginal_cost,
            market_bid_cost=marg_cluster_properties.market_bid_cost,
            startup_cost=base_startup_cost * perturbation,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_startup_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > THERMAL_TEST_REL_ACCURACY


@pytest.mark.parametrize("base_fixed_cost", [250, 500, 1000])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_fixed_cost(
    base_fixed_cost: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Setup thermal cluster properties with base fixed cost
    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=100,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=base_fixed_cost,
        group=ThermalClusterGroup.NUCLEAR,
    )

    # Run base test
    study_name_base = f"e2e_fixedcost_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY
    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        # Test +/-20% fixed cost
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=marg_cluster_properties.marginal_cost,
            market_bid_cost=marg_cluster_properties.market_bid_cost,
            fixed_cost=base_fixed_cost * perturbation,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_fixedcost_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > THERMAL_TEST_REL_ACCURACY


@pytest.mark.parametrize("base_marginal_cost", [1, 10, 100])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_marginal_cost_marketbid_equals(
    base_marginal_cost: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Setup thermal cluster properties
    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=50,
        marginal_cost=base_marginal_cost,
        market_bid_cost=base_marginal_cost,
        group=ThermalClusterGroup.NUCLEAR,
    )

    # Run base test
    study_name_base = f"e2e_marginal_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY
    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        # Test +/-20% marginal cost
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=base_marginal_cost * perturbation,
            market_bid_cost=base_marginal_cost * perturbation,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_marginal_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > THERMAL_TEST_REL_ACCURACY


@pytest.mark.parametrize("base_min_down_time", [2, 4, 6])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_min_down_time(
    base_min_down_time: int,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Ensure min_down_time is < min_up_time for all tests
    base_min_up_time = base_min_down_time + 2

    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=100,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=500,
        min_down_time=base_min_down_time,
        min_up_time=base_min_up_time,
        group=ThermalClusterGroup.NUCLEAR,
    )

    study_name_base = f"e2e_mindown_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY

    for perturbation in [1, -1]:
        perturbed_min_down_time = base_min_down_time + perturbation
        # Always keep min_up_time > min_down_time
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=marg_cluster_properties.marginal_cost,
            market_bid_cost=marg_cluster_properties.market_bid_cost,
            fixed_cost=marg_cluster_properties.fixed_cost,
            min_down_time=perturbed_min_down_time,
            min_up_time=marg_cluster_properties.min_up_time,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_mindown_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > rel_gap_base


@pytest.mark.parametrize("base_min_up_time", [3, 4, 6])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_min_up_time(
    base_min_up_time: int,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Ensure min_down_time < min_up_time for all tests
    base_min_down_time = max(1, base_min_up_time - 1)

    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=100,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=500,
        min_down_time=base_min_down_time,
        min_up_time=base_min_up_time,
        group=ThermalClusterGroup.NUCLEAR,
    )

    study_name_base = f"e2e_minup_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY

    for perturbation in [1, -1]:
        perturbed_min_up_time = base_min_up_time + perturbation
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=marg_cluster_properties.marginal_cost,
            market_bid_cost=marg_cluster_properties.market_bid_cost,
            fixed_cost=marg_cluster_properties.fixed_cost,
            min_down_time=marg_cluster_properties.min_down_time,
            min_up_time=perturbed_min_up_time,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_minup_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > rel_gap_base


@pytest.mark.parametrize("base_min_stable_power", [20, 40, 60])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES)
def test_min_stable_power(
    base_min_stable_power: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Setup thermal cluster properties with base min_stable_power
    marg_cluster_properties = ThermalClusterProperties(
        nominal_capacity=100,
        marginal_cost=20,
        market_bid_cost=20,
        fixed_cost=500,
        startup_cost=1000,
        min_stable_power=base_min_stable_power,
        group=ThermalClusterGroup.NUCLEAR,
    )

    # Run base test
    study_name_base = f"e2e_minstable_base_{str(int(100*time()))}"
    cluster_data_frame_base = pd.DataFrame(
        data=marg_cluster_properties.unit_count
        * marg_cluster_properties.nominal_capacity
        * np.ones((8760, 1))
    )
    createThermalTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        marg_cluster_properties,
        cluster_data_frame_base,
    )
    orig_path_base, conv_path_base = convert_study(
        auto_generated_studies_path, study_name_base, ["thermal"]
    )
    rel_gap_base = first_optim_relgap(
        antares_exec_folder, orig_path_base, conv_path_base, THERMAL_TEST_SOLVER
    )
    assert rel_gap_base < THERMAL_TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        # Test +/-20% min_stable_power
        marg_cluster = ThermalClusterProperties(
            nominal_capacity=marg_cluster_properties.nominal_capacity,
            marginal_cost=marg_cluster_properties.marginal_cost,
            market_bid_cost=marg_cluster_properties.market_bid_cost,
            fixed_cost=marg_cluster_properties.fixed_cost,
            min_stable_power=base_min_stable_power * perturbation,
            group=ThermalClusterGroup.NUCLEAR,
        )
        study_name = f"e2e_minstable_{str(int(100*time()))}"
        cluster_data_frame = pd.DataFrame(
            data=marg_cluster.unit_count
            * marg_cluster.nominal_capacity
            * np.ones((8760, 1))
        )
        createThermalTestAntaresStudy(
            study_name,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            marg_cluster,
            cluster_data_frame,
        )
        orig_path_perturbated = auto_generated_studies_path / study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            orig_path_perturbated,
            conv_path_base,
            THERMAL_TEST_SOLVER,
        )
        assert rel_gap > THERMAL_TEST_REL_ACCURACY
