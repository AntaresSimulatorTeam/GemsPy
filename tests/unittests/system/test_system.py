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

from gems.model.library import Library
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.study.system import Component, System


@pytest.fixture(scope="session")
def libs_dir() -> Path:
    return Path(__file__).parent / "libs"


@pytest.fixture(scope="session")
def lib_dict(libs_dir: Path) -> dict[str, Library]:
    lib_file = libs_dir / "lib.yml"

    with lib_file.open() as f:
        input_lib = parse_yaml_library(f)

    lib_dict = resolve_library([input_lib])
    return lib_dict


def test_system(lib_dict: dict[str, Library]) -> None:
    # This test could be done without parsing the yaml lib, ie. by giving models directly as Python object
    system = System("test")
    assert system.id == "test"
    assert list(system.components) == []
    assert list(system.all_components) == []
    assert list(system.connections) == []

    node_model = lib_dict["basic"].models["basic.node"]

    N1 = Component(model=node_model, id="N1")
    N2 = Component(model=node_model, id="N2")
    system.add_component(N1)
    system.add_component(N2)
    assert list(system.components) == [N1, N2]
    assert system.get_component(N1.id) == N1
    assert system.get_component("N1") == Component(model=node_model, id="N1")
    with pytest.raises(KeyError):
        system.get_component("unknown")
