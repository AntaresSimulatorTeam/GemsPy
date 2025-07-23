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

from gems.model.parsing import InputLibrary, parse_yaml_library
from gems.model.resolve_library import Library, resolve_library
from gems.study.parsing import InputSystem, parse_yaml_components


@pytest.fixture(scope="session")
def libs_dir() -> Path:
    return Path(__file__).parent / "libs"


@pytest.fixture
def systems_dir() -> Path:
    return Path(__file__).parent / "systems"


@pytest.fixture
def series_dir() -> Path:
    return Path(__file__).parent / "series"


@pytest.fixture
def input_system(systems_dir: Path) -> InputSystem:
    compo_file = systems_dir / "system.yml"

    with compo_file.open() as c:
        return parse_yaml_components(c)


@pytest.fixture
def input_library(libs_dir: Path) -> InputLibrary:
    library = libs_dir / "lib_unittest.yml"

    with library.open() as lib:
        return parse_yaml_library(lib)


@pytest.fixture(scope="session")
def lib_dict(libs_dir: Path) -> dict[str, Library]:
    lib_file = libs_dir / "lib.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)

    lib_dict = resolve_library([input_lib])
    return lib_dict
