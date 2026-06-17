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
import re
from pathlib import Path

import pytest

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library

# in following tests "lib_A -> lib_B" means lib_A must be resolved before lib_B


def test_simple_dependency_tree(libs_dir: Path) -> None:
    """basic_lib
        |     |
        V     V
    demand   production
    """
    lib_files = [
        libs_dir / "basic_lib.yml",
        libs_dir / "demand.yml",
        libs_dir / "production.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    lib_dict = resolve_library(input_libs)
    assert len(lib_dict) == 3

    assert len(lib_dict["basic"].models) == 1
    assert len(lib_dict["demand"].models) == 1
    assert len(lib_dict["production"].models) == 1

    assert len(lib_dict["basic"].port_types) == 1
    assert len(lib_dict["demand"].port_types) == 1
    assert len(lib_dict["production"].port_types) == 1

    # changing order in lib_files
    lib_files = [
        libs_dir / "demand.yml",
        libs_dir / "production.yml",
        libs_dir / "basic_lib.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    lib_dict = resolve_library(input_libs)
    assert len(lib_dict["basic"].models) == 1
    assert len(lib_dict["demand"].models) == 1
    assert len(lib_dict["production"].models) == 1

    assert len(lib_dict["basic"].port_types) == 1
    assert len(lib_dict["demand"].port_types) == 1
    assert len(lib_dict["production"].port_types) == 1


def test_multiple_dependencies_tree(libs_dir: Path) -> None:
    """basic_lib   CO2_port
        |     |       |
        V     V       V
    demand   production_with_CO2
    """
    lib_files = [
        libs_dir / "basic_lib.yml",
        libs_dir / "CO2_port.yml",
        libs_dir / "demand.yml",
        libs_dir / "production_with_CO2.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    lib_dict = resolve_library(input_libs)
    assert len(lib_dict["basic"].models) == 1
    assert len(lib_dict["CO2_port"].models) == 0
    assert len(lib_dict["production_CO2"].models) == 1
    assert len(lib_dict["demand"].models) == 1

    assert len(lib_dict["basic"].port_types) == 1
    assert len(lib_dict["CO2_port"].port_types) == 1
    assert len(lib_dict["production_CO2"].port_types) == 2
    assert len(lib_dict["demand"].port_types) == 1


def test_looping_dependency(libs_dir: Path) -> None:
    """looping_lib_1 -> looping_lib_2
    <-
    """
    lib_files = [
        libs_dir / "looping_lib_1.yml",
        libs_dir / "looping_lib_2.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    with pytest.raises(Exception, match=r"Circular import in yaml libraries"):
        resolve_library(input_libs)


def test_model_with_same_name_in_different_lib_ok(libs_dir: Path) -> None:
    """basic_lib   CO2_port
            |     |      |
            V     V      V
    production   production_with_CO2
    """
    lib_files = [
        libs_dir / "basic_lib.yml",
        libs_dir / "CO2_port.yml",
        libs_dir / "production.yml",
        libs_dir / "production_with_CO2.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    lib_dict = resolve_library(input_libs)
    assert len(lib_dict["basic"].models) == 1
    assert len(lib_dict["CO2_port"].models) == 0
    assert len(lib_dict["production_CO2"].models) == 1
    assert len(lib_dict["production"].models) == 1

    assert len(lib_dict["basic"].port_types) == 1
    assert len(lib_dict["CO2_port"].port_types) == 1
    assert len(lib_dict["production_CO2"].port_types) == 2
    assert len(lib_dict["production"].port_types) == 1


def test_model_redefinition_in_same_lib(libs_dir: Path) -> None:
    lib_files = [
        libs_dir / "basic_lib.yml",
        libs_dir / "production_redefinition.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    with pytest.raises(Exception, match=re.escape("Model generator is defined twice")):
        resolve_library(input_libs)


def test_port_redefinition(libs_dir: Path) -> None:
    """basic_lib -> port_redefinition"""
    lib_files = [
        libs_dir / "basic_lib.yml",
        libs_dir / "port_redefinition.yml",
    ]

    input_libs = []
    for lib_file in lib_files:
        with lib_file.open() as f:
            input_libs.append(parse_yaml_library(f))

    with pytest.raises(
        Exception, match=re.escape("Port(s) : {'flow'} is(are) defined twice")
    ):
        lib = resolve_library(input_libs)
