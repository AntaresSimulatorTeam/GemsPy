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

import io
from pathlib import Path
from typing import Type, TypeVar

import pytest

from gems_craft.model.parsing import (
    LibrarySchema,
    parse_yaml_library,
    write_yaml_library,
)
from gems_craft.model.taxonomy import Taxonomy, TaxonomyCategory, TaxonomyItem
from gems_craft.study.parsing import (
    SystemSchema,
    parse_yaml_system,
    write_yaml_system,
)
from gems_craft_hybrid.model.parsing import (
    AreaConnectionSchema,
    HybridLibrarySchema,
    HybridPortTypeSchema,
    PortThermalCapacitySchema,
    parse_yaml_hybrid_library,
)
from gems_craft_hybrid.study.parsing import (
    AreaConnectionsSchema,
    HybridSystemSchema,
    ThermalCapacityConnectionSchema,
    ThermalComponentSchema,
    parse_yaml_hybrid_system,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_hybrid_library(path: Path) -> HybridLibrarySchema:
    with path.open() as f:
        return parse_yaml_hybrid_library(f)


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


# ---------------------------------------------------------------------------
# HybridLibrarySchema — parsing
# ---------------------------------------------------------------------------


def test_load_hybrid_library_parses_area_connection() -> None:
    lib = _load_hybrid_library(FIXTURES / "hybrid_lib.yml")
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert isinstance(flow_port, HybridPortTypeSchema)
    assert flow_port.area_connection == AreaConnectionSchema(
        injection_to_balance="flow",
        spillage_bound="flow",
        unsupplied_energy_bound=None,
    )


def test_load_hybrid_library_port_type_without_area_connection() -> None:
    lib = _load_hybrid_library(FIXTURES / "hybrid_lib.yml")
    signal_port = next(
        pt for pt in lib.port_types if pt.id == "antares_thermal_cluster_capacity"
    )
    assert signal_port.area_connection is None


def test_load_standard_library_on_hybrid_file_raises() -> None:
    """parse_yaml_library (standard) rejects hybrid-only fields due to extra='forbid'."""
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        with pytest.raises(ValueError):
            parse_yaml_library(f)


def test_parse_yaml_hybrid_library_reads_hybrid_fields() -> None:
    with FIXTURES.joinpath("hybrid_lib.yml").open() as f:
        lib = parse_yaml_hybrid_library(f)
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    capacity_port = next(
        pt for pt in lib.port_types if pt.id == "antares_thermal_cluster_capacity"
    )
    assert isinstance(flow_port, HybridPortTypeSchema)
    assert flow_port.area_connection is not None
    assert capacity_port.thermal_capacity_connection is not None
    assert lib == _load_hybrid_library(FIXTURES / "hybrid_lib.yml")


# ---------------------------------------------------------------------------
# HybridLibrarySchema — roundtrip via write_yaml_library
# ---------------------------------------------------------------------------


def test_write_yaml_library_roundtrip_hybrid(tmp_path: Path) -> None:
    original = _load_hybrid_library(FIXTURES / "hybrid_lib.yml")
    out = tmp_path / "lib.yml"
    write_yaml_library(original, out)
    reloaded = _load_hybrid_library(out)
    assert reloaded == original


# ---------------------------------------------------------------------------
# HybridLibrarySchema — thermal-capacity-connections
# ---------------------------------------------------------------------------


def test_load_hybrid_library_parses_thermal_capacity_connection() -> None:
    lib = _load_hybrid_library(FIXTURES / "hybrid_lib.yml")
    capacity_port = next(
        pt for pt in lib.port_types if pt.id == "antares_thermal_cluster_capacity"
    )
    assert isinstance(capacity_port, HybridPortTypeSchema)
    assert capacity_port.thermal_capacity_connection == PortThermalCapacitySchema(
        capacity_field="capacity"
    )


def test_load_hybrid_library_port_type_without_thermal_capacity_connection() -> None:
    lib = _load_hybrid_library(FIXTURES / "hybrid_lib.yml")
    flow_port = next(pt for pt in lib.port_types if pt.id == "flow")
    assert flow_port.thermal_capacity_connection is None


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


# ---------------------------------------------------------------------------
# Taxonomy check is threaded through the hybrid parser
# ---------------------------------------------------------------------------


_TAXONOMY_LIB = """
library:
  id: mylib
  taxonomy: hybrid_taxonomy
  port-types:
    - id: flow
      fields:
        - id: flow
  models:
    - id: generator
      taxonomy-category: production
      ports:
        - id: injection_port
          type: flow
"""


def _taxonomy_requiring(port_id: str) -> Taxonomy:
    return Taxonomy(
        id="hybrid_taxonomy",
        categories=[
            TaxonomyCategory(id="production", ports=[TaxonomyItem(id=port_id)])
        ],
    )


def test_parse_hybrid_library_checks_declared_taxonomy() -> None:
    lib = parse_yaml_hybrid_library(
        io.StringIO(_TAXONOMY_LIB), _taxonomy_requiring("injection_port")
    )
    assert lib.taxonomy == "hybrid_taxonomy"


def test_parse_hybrid_library_raises_on_taxonomy_violation() -> None:
    with pytest.raises(ValueError, match="missing_port"):
        parse_yaml_hybrid_library(
            io.StringIO(_TAXONOMY_LIB), _taxonomy_requiring("missing_port")
        )
