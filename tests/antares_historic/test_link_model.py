from pathlib import Path
from time import time

import numpy as np
import pytest

from tests.antares_historic.utils import (
    convert_study,
    createLinkTestAntaresStudy,
    first_optim_relgap,
)

LOAD_FILES_DIR = Path("tests/antares_historic/data")
LINK_TEST_REL_ACCURACY = 1e-6
LINK_TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2


def link_test_procedure(
    study_name: str,
    study_path: Path,
    link_capacity_direct: np.ndarray,
    link_capacity_indirect: np.ndarray,
    load1_time_serie_file: Path,
    load2_time_serie_file: Path,
    exec_folder: Path,
) -> None:
    createLinkTestAntaresStudy(
        study_name,
        study_path,
        load1_time_serie_file,
        load2_time_serie_file,
        link_capacity_direct,
        link_capacity_indirect,
    )
    original_study_path, converted_study_path = convert_study(
        study_path, study_name, ["link", "thermal"]
    )
    rel_gap = first_optim_relgap(
        exec_folder, original_study_path, converted_study_path, LINK_TEST_SOLVER
    )
    assert rel_gap < LINK_TEST_REL_ACCURACY


@pytest.mark.parametrize("capacity_direct", [10, 100])
@pytest.mark.parametrize("capacity_indirect", [10, 50])
def test_general_link(
    capacity_direct: float,
    capacity_indirect: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    study_name = f"link_test_study_{str(int(100*time()))}"
    load1_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    load2_time_serie_file = LOAD_FILES_DIR / "load_matrix_2.txt"
    link_capacity_direct = capacity_direct * np.ones((8760, 1))
    link_capacity_indirect = capacity_indirect * np.ones((8760, 1))
    link_test_procedure(
        study_name,
        auto_generated_studies_path,
        link_capacity_direct,
        link_capacity_indirect,
        load1_time_serie_file,
        load2_time_serie_file,
        antares_exec_folder,
    )
    study_name = f"link_test_study_{str(int(100*time()))}"
    link_capacity_direct = capacity_direct * np.random.random((8760, 1))
    link_capacity_indirect = capacity_indirect * np.random.random((8760, 1))

    link_test_procedure(
        study_name,
        auto_generated_studies_path,
        link_capacity_direct,
        link_capacity_indirect,
        load1_time_serie_file,
        load2_time_serie_file,
        antares_exec_folder,
    )


@pytest.mark.parametrize("capacity_direct", [10, 100])
def test_direct_capacity(
    capacity_direct: float, auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    capacity_indirect = 50.0
    study_name = f"link_test_study_{str(int(100*time()))}"
    load1_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    load2_time_serie_file = LOAD_FILES_DIR / "load_matrix_2.txt"
    link_capacity_direct = capacity_direct * np.ones((8760, 1))
    link_capacity_indirect = capacity_indirect * np.ones((8760, 1))

    createLinkTestAntaresStudy(
        study_name,
        auto_generated_studies_path,
        load1_time_serie_file,
        load2_time_serie_file,
        link_capacity_direct,
        link_capacity_indirect,
    )

    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name, ["link"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, LINK_TEST_SOLVER
    )
    assert rel_gap < LINK_TEST_REL_ACCURACY

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"link_test_study_{str(int(100*time()))}"
        link_capacity_direct_perturbed = (
            capacity_direct * modification * np.ones((8760, 1))
        )
        createLinkTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load1_time_serie_file,
            load2_time_serie_file,
            link_capacity_direct_perturbed,
            link_capacity_indirect,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            perturbed_study_path,
            converted_study_path,
            LINK_TEST_SOLVER,
        )
        assert rel_gap > 10 * LINK_TEST_REL_ACCURACY


@pytest.mark.parametrize("capacity_indirect", [20, 100])
def test_indirect_capacity(
    capacity_indirect: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    capacity_direct = 50.0
    study_name = f"link_test_study_{str(int(100*time()))}"
    load1_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    load2_time_serie_file = LOAD_FILES_DIR / "load_matrix_2.txt"
    link_capacity_direct = capacity_direct * np.ones((8760, 1))
    link_capacity_indirect = capacity_indirect * np.ones((8760, 1))

    createLinkTestAntaresStudy(
        study_name,
        auto_generated_studies_path,
        load1_time_serie_file,
        load2_time_serie_file,
        link_capacity_direct,
        link_capacity_indirect,
    )

    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name, ["link"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, LINK_TEST_SOLVER
    )
    assert rel_gap < LINK_TEST_REL_ACCURACY

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"link_test_study_{str(int(100*time()))}"
        link_capacity_indirect_perturbed = (
            capacity_indirect * modification * np.ones((8760, 1))
        )
        createLinkTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load1_time_serie_file,
            load2_time_serie_file,
            link_capacity_direct,
            link_capacity_indirect_perturbed,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            perturbed_study_path,
            converted_study_path,
            LINK_TEST_SOLVER,
        )
        assert rel_gap > 10 * LINK_TEST_REL_ACCURACY
