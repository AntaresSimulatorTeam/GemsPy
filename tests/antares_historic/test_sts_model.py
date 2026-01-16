import os
from dataclasses import replace
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pytest
from antares.craft import STStorageProperties

from gems.input_converter.src.logger import Logger
from tests.antares_historic.utils import (
    convert_study,
    createSTSTestAntaresStudy,
    first_optim_relgap,
)

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "SHORT TERM STORAGE"  : CONSTANT DATA  ##

LOAD_FILES_DIR = Path("tests/antares_historic/data")
STS_TEST_REL_ACCURACY = 1e-6
STS_TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2
LOAD_TIME_SERIE_FILES_STS = [
    "load_matrix_1.txt",
    # "load_matrix_2.txt", #uncomment to test with different load profile
    # "load_matrix_original.txt",
]

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "SHORT TERM STORAGE"  : TESTS  ##

# General tests [OK:sts_list_general_test]

# Testing Boolean/discrete parameters
## initial_level_optim [TODO]
## enabled [TODO]
## penalize_variation_injection [TODO]
## penalize_variation_withdrawal [TODO]

# Testing Float parameters
## injection_nominal_capacity [OK : test_injection_nominal_capacity]
## withdrawal_nominal_capacity [OK: test_withdrawal_nominal_capacity]
## reservoir_capacity [OK: test_reservoir_capacity]
## efficiency [OK : test_efficiency]
## efficiency_withdrawal [OK : test_efficiency_withdrawal]
## initial_level [OK : test_initial_level]

# Testing Timeseries parameters
## p_max_injection [TODO]
## p_max_withdrawal [TODO]
## lower_rule_curve [TODO]
## upper_rule_curve [TODO]
## storage_inflows [TODO]
## cost_injection [TODO]
## cost_withdrawal [TODO]
## cost_level [TODO]
## cost_variation_injection [TODO]
## cost_variation_withdrawal [TODO]


def sts_test_procedure(
    study_name: str,
    study_path: Path,
    sts_properties: STStorageProperties,
    load_time_serie_file: Path,
    exec_folder: Path,
) -> None:
    createSTSTestAntaresStudy(
        study_name,
        study_path,
        load_time_serie_file,
        sts_properties,
    )
    original_study_path, converted_study_path = convert_study(
        study_path, study_name, ["short-term-storage"]
    )
    rel_gap = first_optim_relgap(
        exec_folder, original_study_path, converted_study_path, STS_TEST_SOLVER
    )
    assert rel_gap < STS_TEST_REL_ACCURACY


def sts_test_procedure_float_param(
    sts_properties: STStorageProperties,
    tested_param: str,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Run base test
    study_name_base = f"e2e_{str(int(100*time()))}"

    createSTSTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        sts_properties,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["short-term-storage"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, STS_TEST_SOLVER
    )
    assert rel_gap < STS_TEST_REL_ACCURACY

    ref_value_param = getattr(sts_properties, tested_param)

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        sts_properties_perturbated = replace(
            sts_properties, **{tested_param: ref_value_param * modification}
        )

        study_name_perturbated = f"e2e_{str(int(100*time()))}"

        createSTSTestAntaresStudy(
            study_name_perturbated,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            sts_properties_perturbated,
        )
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            auto_generated_studies_path / study_name_perturbated,
            converted_study_path,
            STS_TEST_SOLVER,
        )
        assert rel_gap > 100 * STS_TEST_REL_ACCURACY


@pytest.fixture(scope="session")
def sts_list_general_test() -> list[STStorageProperties]:
    return [
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency=1,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency=0.8,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=100,
            efficiency=0.8,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=100,
            efficiency=0.8,
            initial_level=0.2,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency_withdrawal=0.9,
            efficiency=0.85,
            initial_level=0.5,
        ),
    ]  # ,


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_general_sts(
    sts_list_general_test: list[STStorageProperties],
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    for sts_properties in sts_list_general_test:
        study_name = f"e2e_general_test_{str(int(100*time()))}"
        sts_test_procedure(
            study_name,
            auto_generated_studies_path,
            sts_properties,
            LOAD_FILES_DIR / load_time_serie_file,
            antares_exec_folder,
        )


@pytest.mark.parametrize("injection_nominal_capacity_base", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_injection_nominal_capacity(
    injection_nominal_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=injection_nominal_capacity_base,
        withdrawal_nominal_capacity=200,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "injection_nominal_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("withdrawal_nominal_capacity_base", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_withdrawal_nominal_capacity(
    withdrawal_nominal_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=200,
        withdrawal_nominal_capacity=withdrawal_nominal_capacity_base,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "withdrawal_nominal_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("reservoir_capacity_base", [100.0, 200.0, 300.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_reservoir_capacity(
    reservoir_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=reservoir_capacity_base,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "reservoir_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("efficiency_base", [0.5, 0.6, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_efficiency(
    efficiency_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=200,
        efficiency=efficiency_base,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "efficiency",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("efficiency_withdrawal_base", [0.5, 0.6, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_efficiency_withdrawal(
    efficiency_withdrawal_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=200,
        efficiency_withdrawal=efficiency_withdrawal_base,
        efficiency=0.5 / MODIFICATION_RATIO,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "efficiency_withdrawal",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("initial_level_base", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_initial_level(
    initial_level_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=250,
        initial_level=initial_level_base,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "initial_level",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )
