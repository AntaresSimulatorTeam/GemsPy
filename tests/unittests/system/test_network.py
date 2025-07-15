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
from gems.study.network import Network, Node


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


def test_network(lib_dict: dict[str, Library]) -> None:
    # This test could be done without parsing the yaml lib, ie. by giving models directly as Python object
    network = Network("test")
    assert network.id == "test"
    assert list(network.nodes) == []
    assert list(network.components) == []
    assert list(network.all_components) == []
    assert list(network.connections) == []

    with pytest.raises(KeyError):
        network.get_node("N")

    node_model = lib_dict["basic"].models["node"]

    N1 = Node(model=node_model, id="N1")
    N2 = Node(model=node_model, id="N2")
    network.add_node(N1)
    network.add_node(N2)
    assert list(network.nodes) == [N1, N2]
    assert network.get_node(N1.id) == N1
    assert network.get_component("N1") == Node(model=node_model, id="N1")
    with pytest.raises(KeyError):
        network.get_component("unknown")
