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

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import Library, resolve_library


@pytest.fixture(scope="session")
def libs_dir() -> Path:
    return Path(__file__).parent / "libs"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parents[2] / "data"


@pytest.fixture(scope="session")
def lib_dict(libs_dir: Path) -> dict[str, Library]:
    lib_file = libs_dir / "lib_unittest.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)

    lib_dict = resolve_library([input_lib])
    return lib_dict


@pytest.fixture(scope="session")
def lib_dict_sc(libs_dir: Path) -> dict[str, Library]:
    lib_sc_file = libs_dir / "standard_sc.yml"

    with lib_sc_file.open() as f:
        input_lib_sc = parse_yaml_library(f)

    lib_dict_sc = resolve_library([input_lib_sc])
    return lib_dict_sc
