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

"""Tests for gems_craft_hybrid parsing: HybridSystemSchema and HybridLibrarySchema."""

from pathlib import Path
from typing import Type, TypeVar

import pytest

from gems_craft.model.parsing import (
    LibrarySchema,
    parse_yaml_library,
    write_yaml_library,
)
from gems_craft.study.parsing import (
    SystemSchema,
    parse_yaml_components,
    write_yaml_system,
)
from gems_craft_hybrid.model.parsing import (
    AreaConnectionSchema,
    HybridLibrarySchema,
    HybridPortTypeSchema,
    PortThermalCapacitySchema,
)
from gems_craft_hybrid.study.parsing import (
    AreaConnectionsSchema,
    HybridSystemSchema,
    ThermalCapacityConnectionSchema,
    ThermalComponentSchema,
)

FIXTURES = Path(__file__).parent / "fixtures"

_L = TypeVar("_L", bound=LibrarySchema)
_S = TypeVar("_S", bound=SystemSchema)


def _load_library(path: Path, schema: Type[_L]) -> _L:
    with path.open() as f:
        return parse_yaml_library(f, schema)


def _load_system(path: Path, schema: Type[_S]) -> _S:
    with path.open() as f:
        return parse_yaml_components(f, schema)


# ---------------------------------------------------------------------------
# HybridSystemSchema — parsing
# ---------------------------------------------------------------------------


def test_load_hybrid_system_parses_area_connections() -> None:
    system = _load_system(FIXTURES / "hybrid_system.yml", HybridSystemSchema)
    assert isinstance(system, HybridSystemSchema)
    assert system.area_connections is not None
    assert len(system.area_connections) == 1
    assert system.area_connections[0] == AreaConnectionsSchema(
        component="G", port="injection_port", area="fr"
    )


def test_load_hybrid_system_parses_standard_fields() -> None:
    system = _load_system(FIXTURES / "hybrid_system.yml", HybridSystemSchema)
    assert len(system.components) == 2
    assert system.connections is not None
    assert len(system.connections) == 1


def test_load_standard_system_on_hybrid_file_raises() -> None:
    """parse_yaml_components (standard) rejects hybrid-only fields due to extra='forbid'."""
    with FIXTURES.joinpath("hybrid_system.yml").open() as f:
        with pytest.raises(ValueError):
            parse_yaml_components(f)


# ---------------------------------------------------------------------------
# HybridSystemSchema — roundtrip via write_yaml_system
# ---------------------------------------------------------------------------


def test_write_yaml_system_roundtrip_hybrid(tmp_path: Path) -> None:
    original = _load_system(FIXTURES / "hybrid_system.yml", HybridSystemSchema)
    out = tmp_path / "system.yml"
    write_yaml_system(original, out)
    reloaded = _load_system(out, HybridSystemSchema)
    assert reloaded == original


# ---------------------------------------------------------------------------
# HybridLibrarySchema — parsing
# ---------------------------------------------------------------------------


def test_load_hybrid_library_parses_area_connection() -> None:
    lib = _load_library(FIXTURES / "hybrid_lib.yml", HybridLibrarySchema)
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert isinstance(flow_port, HybridPortTypeSchema)
    assert flow_port.area_connection == AreaConnectionSchema(
        injection_to_balance="flow",
        spillage_bound="flow",
        unsupplied_energy_bound=None,
    )


def test_load_hybrid_library_port_type_without_area_connection() -> None:
    lib = _load_library(FIXTURES / "hybrid_lib.yml", HybridLibrarySchema)
    signal_port = next(pt for pt in lib.port_types if pt.id == "signal")
    assert signal_port.area_connection is None


def test_load_standard_library_on_hybrid_file_raises() -> None:
    """parse_yaml_library (standard) rejects hybrid-only fields due to extra='forbid'."""
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        with pytest.raises(ValueError):
            parse_yaml_library(f)


# ---------------------------------------------------------------------------
# HybridLibrarySchema — roundtrip via write_yaml_library
# ---------------------------------------------------------------------------


def test_write_yaml_library_roundtrip_hybrid(tmp_path: Path) -> None:
    original = _load_library(FIXTURES / "hybrid_lib.yml", HybridLibrarySchema)
    out = tmp_path / "lib.yml"
    write_yaml_library(original, out)
    reloaded = _load_library(out, HybridLibrarySchema)
    assert reloaded == original


# ---------------------------------------------------------------------------
# HybridLibrarySchema — thermal-capacity-connections
# ---------------------------------------------------------------------------


def test_load_hybrid_library_parses_thermal_capacity_connection() -> None:
    lib = _load_library(FIXTURES / "hybrid_lib.yml", HybridLibrarySchema)
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert isinstance(flow_port, HybridPortTypeSchema)
    assert flow_port.thermal_capacity_connection == PortThermalCapacitySchema(
        capacity_field="flow"
    )


def test_load_hybrid_library_port_type_without_thermal_capacity_connection() -> None:
    lib = _load_library(FIXTURES / "hybrid_lib.yml", HybridLibrarySchema)
    signal_port = next(pt for pt in lib.port_types if pt.id == "signal")
    assert signal_port.thermal_capacity_connection is None


# ---------------------------------------------------------------------------
# HybridSystemSchema — thermal-capacity-connections
# ---------------------------------------------------------------------------


def test_load_hybrid_system_parses_thermal_capacity_connections() -> None:
    system = _load_system(FIXTURES / "hybrid_system.yml", HybridSystemSchema)
    assert system.thermal_capacity_connections is not None
    assert len(system.thermal_capacity_connections) == 1
    conn = system.thermal_capacity_connections[0]
    assert isinstance(conn, ThermalCapacityConnectionSchema)
    assert conn.component == "G"
    assert conn.port == "injection_port"
    assert conn.thermal_component == ThermalComponentSchema(
        area="fr", cluster_id="nuclear1"
    )
