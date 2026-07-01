# Copyright (c) 2024, RTE (https://www.rte-france.com)
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
from pathlib import Path

import pytest

from gems_craft.model.parsing import load_yaml_library
from gems_runner.model.resolve_library import Library, resolve_library


@pytest.fixture(scope="session")
def libs_dir() -> Path:
    return Path(__file__).parent / "libs"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parents[2] / "data"


@pytest.fixture(scope="session")
def lib_dict(libs_dir: Path) -> dict[str, Library]:
    input_lib = load_yaml_library(libs_dir / "lib_unittest.yml")
    return resolve_library([input_lib])


@pytest.fixture(scope="session")
def lib_dict_sc(libs_dir: Path) -> dict[str, Library]:
    input_lib_sc = load_yaml_library(libs_dir / "standard_sc.yml")
    return resolve_library([input_lib_sc])
