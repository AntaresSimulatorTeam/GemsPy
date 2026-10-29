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

"""Tests for gems_craft_hybrid parsing: HybridSystemSchema, and the hybrid-only
area-connection / thermal-capacity-connection fields on the base PortTypeSchema."""

from pathlib import Path
from typing import Any, Dict

import pytest
from yaml import safe_load

from gems_craft.model.parsing import (
    AreaConnectionSchema,
    FieldSchema,
    LibrarySchema,
    PortThermalCapacitySchema,
    PortTypeSchema,
    parse_yaml_library,
    write_yaml_library,
)
from gems_craft.study.parsing import parse_yaml_system, write_yaml_system
from gems_craft_hybrid.study.parsing import (
    AreaConnectionsSchema,
    HybridSystemSchema,
    ThermalCapacityConnectionSchema,
    ThermalComponentSchema,
    parse_yaml_hybrid_system,
)

FIXTURES = Path(__file__).parent / "fixtures"
# Full GEMS libraries, without any hybrid-only field
FULL_GEMS_LIBS = Path(__file__).parents[1] / "gems_craft" / "lib_parsing" / "libs"


def _load_hybrid_system(path: Path) -> HybridSystemSchema:
    with path.open() as f:
        return parse_yaml_hybrid_system(f)


# ---------------------------------------------------------------------------
# HybridSystemSchema — parsing
# ---------------------------------------------------------------------------


def test_load_hybrid_system_parses_area_connections() -> None:
    system = _load_hybrid_system(FIXTURES / "hybrid_system.yml")
    assert isinstance(system, HybridSystemSchema)
    assert system.area_connections is not None
    assert len(system.area_connections) == 1
    assert system.area_connections[0] == AreaConnectionsSchema(
        component="G2", port="injection_port", area="fr"
    )


def test_load_hybrid_system_parses_standard_fields() -> None:
    system = _load_hybrid_system(FIXTURES / "hybrid_system.yml")
    assert len(system.components) == 4
    assert system.connections is not None
    assert len(system.connections) == 1


def test_load_standard_system_on_hybrid_file_raises() -> None:
    """parse_yaml_system (standard) rejects hybrid-only fields due to extra='forbid'."""
    with FIXTURES.joinpath("hybrid_system.yml").open() as f:
        with pytest.raises(ValueError):
            parse_yaml_system(f)


def test_parse_yaml_hybrid_system_reads_hybrid_fields() -> None:
    with FIXTURES.joinpath("hybrid_system.yml").open() as f:
        system = parse_yaml_hybrid_system(f)
    assert isinstance(system, HybridSystemSchema)
    assert system.area_connections is not None
    assert system.thermal_capacity_connections is not None
    assert system == _load_hybrid_system(FIXTURES / "hybrid_system.yml")


# ---------------------------------------------------------------------------
# HybridSystemSchema — roundtrip via write_yaml_system
# ---------------------------------------------------------------------------


def test_write_yaml_system_roundtrip_hybrid(tmp_path: Path) -> None:
    original = _load_hybrid_system(FIXTURES / "hybrid_system.yml")
    out = tmp_path / "system.yml"
    write_yaml_system(original, out)
    reloaded = _load_hybrid_system(out)
    assert reloaded == original


def test_write_yaml_system_without_hybrid_entries_writes_no_hybrid_key(
    tmp_path: Path,
) -> None:
    system = HybridSystemSchema(id="full_gems")
    out = tmp_path / "system.yml"
    write_yaml_system(system, out)
    written = out.read_text()
    assert "area-connections" not in written
    assert "thermal-capacity-connections" not in written


# ---------------------------------------------------------------------------
# PortTypeSchema — area-connection / thermal-capacity-connection parsing
# ---------------------------------------------------------------------------


def test_load_library_parses_area_connection() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_library(f)
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert flow_port.area_connection == AreaConnectionSchema(
        injection_to_balance="flow",
        spillage_bound="flow",
        unsupplied_energy_bound=None,
    )


def test_load_library_parses_area_connection_price() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_library(f)
    port = next(pt for pt in lib.port_types if pt.id == "flow_with_price")
    assert port.area_connection == AreaConnectionSchema(
        injection_to_balance="flow",
        spillage_bound=None,
        unsupplied_energy_bound=None,
        price="price",
    )


def test_load_library_port_type_without_area_connection() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_library(f)
    signal_port = next(
        pt for pt in lib.port_types if pt.id == "antares_thermal_cluster_capacity"
    )
    assert signal_port.area_connection is None


def test_load_library_parses_thermal_capacity_connection() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_library(f)
    capacity_port = next(
        pt for pt in lib.port_types if pt.id == "antares_thermal_cluster_capacity"
    )
    assert capacity_port.thermal_capacity_connection == PortThermalCapacitySchema(
        capacity_field="capacity"
    )


def test_load_library_port_type_without_thermal_capacity_connection() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_library(f)
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert flow_port.thermal_capacity_connection is None


# ---------------------------------------------------------------------------
# LibrarySchema — roundtrip via write_yaml_library (hybrid fields)
# ---------------------------------------------------------------------------


def test_write_yaml_library_roundtrip_hybrid(tmp_path: Path) -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        original = parse_yaml_library(f)
    out = tmp_path / "lib.yml"
    write_yaml_library(original, out)
    with out.open() as f:
        reloaded = parse_yaml_library(f)
    assert reloaded == original


def _written_area_connections(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open() as f:
        tree = safe_load(f)
    return {
        port_type["id"]: port_type["area-connection"]
        for port_type in tree["library"]["port-types"]
        if "area-connection" in port_type
    }


def test_write_yaml_library_keeps_empty_area_connection_keys(tmp_path: Path) -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        library = parse_yaml_library(f)
    out = tmp_path / "lib.yml"
    write_yaml_library(library, out)
    written = _written_area_connections(out)
    # Mandatory keys first, in field order, then the optional price key
    assert list(written["flow_with_price"]) == [
        "injection-to-balance",
        "spillage-bound",
        "unsupplied-energy-bound",
        "price",
    ]
    assert written == {
        "flow": {
            "injection-to-balance": "flow",
            "spillage-bound": "flow",
            "unsupplied-energy-bound": None,
        },
        "flow_with_price": {
            "injection-to-balance": "flow",
            "spillage-bound": None,
            "unsupplied-energy-bound": None,
            "price": "price",
        },
    }


def test_write_yaml_library_writes_mandatory_area_connection_keys_of_python_library(
    tmp_path: Path,
) -> None:
    library = LibrarySchema(
        id="lib",
        port_types=[
            PortTypeSchema(
                id="flow",
                fields=[FieldSchema(id="flow")],
                area_connection=AreaConnectionSchema(injection_to_balance="flow"),
            ),
            PortTypeSchema(id="plain", fields=[FieldSchema(id="flow")]),
        ],
    )
    out = tmp_path / "lib.yml"
    write_yaml_library(library, out)
    assert _written_area_connections(out) == {
        "flow": {
            "injection-to-balance": "flow",
            "spillage-bound": None,
            "unsupplied-energy-bound": None,
        }
    }


@pytest.mark.parametrize(
    "library_file",
    sorted(FULL_GEMS_LIBS.glob("*.yml")),
    ids=lambda path: path.name,
)
def test_write_yaml_library_of_full_gems_library_writes_no_hybrid_key(
    library_file: Path, tmp_path: Path
) -> None:
    with library_file.open() as f:
        library = parse_yaml_library(f)
    out = tmp_path / "lib.yml"
    write_yaml_library(library, out)
    written = out.read_text()
    assert "area-connection" not in written
    assert "thermal-capacity-connection" not in written


def test_write_yaml_library_of_python_full_gems_library_writes_no_hybrid_key(
    tmp_path: Path,
) -> None:
    library = LibrarySchema(
        id="lib",
        port_types=[PortTypeSchema(id="flow", fields=[FieldSchema(id="flow")])],
    )
    out = tmp_path / "lib.yml"
    write_yaml_library(library, out)
    written = out.read_text()
    assert "area-connection" not in written
    assert "thermal-capacity-connection" not in written


# ---------------------------------------------------------------------------
# HybridSystemSchema — thermal-capacity-connections
# ---------------------------------------------------------------------------


def test_load_hybrid_system_parses_thermal_capacity_connections() -> None:
    system = _load_hybrid_system(FIXTURES / "hybrid_system.yml")
    assert system.thermal_capacity_connections is not None
    assert len(system.thermal_capacity_connections) == 1
    conn = system.thermal_capacity_connections[0]
    assert isinstance(conn, ThermalCapacityConnectionSchema)
    assert conn.component == "I"
    assert conn.port == "investment_port"
    assert conn.thermal_component == ThermalComponentSchema(
        area="fr", cluster_id="nuclear1"
    )
