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

"""Tests for gems_craft write functions: system, library, optim-config, scenario builder."""

from pathlib import Path

import numpy as np
import pytest

from gems_craft.model.parsing import (
    LibrarySchema,
    load_yaml_library,
    write_yaml_library,
)
from gems_craft.optim_config import load_yaml_optim_config, write_yaml_optim_config
from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionConfig,
    ResolutionMode,
    ScenarioScopeConfig,
    SolverOptionsConfig,
    TimeScopeConfig,
)
from gems_craft.study.parsing import SystemSchema, load_yaml_system, write_yaml_system
from gems_craft.study.scenario_builder import ScenarioBuilder

# ---------------------------------------------------------------------------
# write_yaml_system
# ---------------------------------------------------------------------------


def test_write_yaml_system_roundtrip(tmp_path: Path) -> None:
    """A SystemSchema written then re-parsed must equal the original."""
    original_path = (
        Path(__file__).parent.parent / "system_parsing" / "systems" / "system.yml"
    )
    original = load_yaml_system(original_path)

    out = tmp_path / "system.yml"
    write_yaml_system(original, out)

    reloaded = load_yaml_system(out)
    assert reloaded == original


def test_write_yaml_system_creates_parent_dirs(tmp_path: Path) -> None:
    """write_yaml_system creates missing parent directories."""
    system = SystemSchema(components=[], connections=None)
    out = tmp_path / "sub" / "dir" / "system.yml"
    write_yaml_system(system, out)
    assert out.exists()


def test_write_yaml_system_uses_kebab_keys(tmp_path: Path) -> None:
    """Output YAML must use kebab-case keys (e.g. 'time-dependent', not 'time_dependent')."""
    from gems_craft.study.parsing import ComponentParameterSchema, ComponentSchema

    system = SystemSchema(
        components=[
            ComponentSchema(
                id="gen",
                model="lib.generator",
                parameters=[
                    ComponentParameterSchema(
                        id="cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        value=30.0,
                    )
                ],
            )
        ]
    )
    out = tmp_path / "system.yml"
    write_yaml_system(system, out)
    content = out.read_text()
    assert "time-dependent" in content
    assert "time_dependent" not in content


# ---------------------------------------------------------------------------
# write_yaml_library
# ---------------------------------------------------------------------------


def test_write_yaml_library_roundtrip(tmp_path: Path) -> None:
    """A LibrarySchema written then re-parsed must equal the original."""
    original_path = (
        Path(__file__).parent.parent / "lib_parsing" / "libs" / "basic_lib.yml"
    )
    original = load_yaml_library(original_path)

    out = tmp_path / "lib.yml"
    write_yaml_library(original, out)

    reloaded = load_yaml_library(out)
    assert reloaded == original


def test_write_yaml_library_creates_parent_dirs(tmp_path: Path) -> None:
    """write_yaml_library creates missing parent directories."""
    library = LibrarySchema(id="empty")
    out = tmp_path / "sub" / "lib.yml"
    write_yaml_library(library, out)
    assert out.exists()


def test_write_yaml_library_uses_kebab_keys(tmp_path: Path) -> None:
    """Output YAML must use kebab-case keys (e.g. 'port-types', not 'port_types')."""
    from gems_craft.model.parsing import FieldSchema, PortTypeSchema

    library = LibrarySchema(
        id="mylib",
        port_types=[PortTypeSchema(id="flow", fields=[FieldSchema(id="flow")])],
    )
    out = tmp_path / "lib.yml"
    write_yaml_library(library, out)
    content = out.read_text()
    assert "port-types" in content
    assert "port_types" not in content


# ---------------------------------------------------------------------------
# write_yaml_optim_config
# ---------------------------------------------------------------------------


def test_write_yaml_optim_config_roundtrip(tmp_path: Path) -> None:
    """An OptimConfig written then re-loaded must equal the original."""
    config = OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=0, last_time_step=8759),
        solver_options=SolverOptionsConfig(name="highs", logs=False),
        scenario_scope=ScenarioScopeConfig(include=["0-2"]),
        resolution=ResolutionConfig(
            mode=ResolutionMode.SEQUENTIAL_SUBPROBLEMS, block_length=168
        ),
    )
    out = tmp_path / "optim-config.yml"
    write_yaml_optim_config(config, out)

    reloaded = load_yaml_optim_config(out)
    assert reloaded is not None
    assert reloaded.time_scope.last_time_step == 8759
    assert reloaded.resolution.mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS
    assert reloaded.resolution.block_length == 168
    assert reloaded.scenario_scope.scenario_ids == [0, 1, 2]


def test_write_yaml_optim_config_creates_parent_dirs(tmp_path: Path) -> None:
    """write_yaml_optim_config creates missing parent directories."""
    config = OptimConfig()
    out = tmp_path / "sub" / "optim-config.yml"
    write_yaml_optim_config(config, out)
    assert out.exists()


def test_write_yaml_optim_config_uses_kebab_keys(tmp_path: Path) -> None:
    """Output YAML must use kebab-case keys."""
    config = OptimConfig(
        time_scope=TimeScopeConfig(first_time_step=10, last_time_step=100),
    )
    out = tmp_path / "optim-config.yml"
    write_yaml_optim_config(config, out)
    content = out.read_text()
    assert "time-scope" in content
    assert "time_scope" not in content
    assert "first-time-step" in content


# ---------------------------------------------------------------------------
# ScenarioBuilder.write_dat / load_dat
# ---------------------------------------------------------------------------


def test_scenario_builder_write_dat_roundtrip(tmp_path: Path) -> None:
    """A ScenarioBuilder dumped then re-loaded must produce identical mappings."""
    sb = ScenarioBuilder(
        _group_arrays={
            "load": np.array([0, 1, 0, 1]),
            "cost-group": np.array([0, 0, 1, 1]),
        }
    )
    out = tmp_path / "modeler-scenariobuilder.dat"
    sb.write_dat(out)

    reloaded = ScenarioBuilder.load_dat(out)

    mc = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(
        sb.resolve_vectorized("load", mc),
        reloaded.resolve_vectorized("load", mc),
    )
    np.testing.assert_array_equal(
        sb.resolve_vectorized("cost-group", mc),
        reloaded.resolve_vectorized("cost-group", mc),
    )


def test_scenario_builder_write_dat_format(tmp_path: Path) -> None:
    """Written file uses 1-based column indices and correct line format."""
    sb = ScenarioBuilder(_group_arrays={"wind": np.array([0, 2])})
    out = tmp_path / "sb.dat"
    sb.write_dat(out)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines[0] == "wind, 0 = 1"  # col_idx 0 → 1-based = 1
    assert lines[1] == "wind, 1 = 3"  # col_idx 2 → 1-based = 3


def test_scenario_builder_write_dat_creates_parent_dirs(tmp_path: Path) -> None:
    """ScenarioBuilder.write_dat creates missing parent directories."""
    sb = ScenarioBuilder()
    out = tmp_path / "sub" / "modeler-scenariobuilder.dat"
    sb.write_dat(out)
    assert out.exists()


def test_scenario_builder_write_dat_load_existing_fixture(tmp_path: Path) -> None:
    """Load an existing fixture, write it, reload it, verify identity."""
    fixture = (
        Path(__file__).parent.parent
        / "scenario_builder"
        / "series"
        / "modeler-scenariobuilder.dat"
    )
    original = ScenarioBuilder.load_dat(fixture)
    out = tmp_path / "modeler-scenariobuilder.dat"
    original.write_dat(out)
    reloaded = ScenarioBuilder.load_dat(out)

    mc = np.array([0, 1, 2, 3])
    for group in ("load", "cost-group"):
        np.testing.assert_array_equal(
            original.resolve_vectorized(group, mc),
            reloaded.resolve_vectorized(group, mc),
        )
