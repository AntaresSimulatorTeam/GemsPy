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

"""Shared fixtures/helpers for the thermal-heuristic e2e tests.

Instead of committing one full study directory per (case, mode) combination
(milp/lp/accurate/fast), each case commits a single base study (the milp
form). The other modes are derived in memory at test time:

- lp / accurate / fast are obtained by mutating the parsed ``SystemSchema``'s
  ``integer-strategy`` field on the thermal component(s) — the production
  code (``optimization.py:_create_variables_for_model``) already relaxes any
  component whose strategy is ``relaxed`` or ``heuristic`` to continuous
  variables at problem-build time, regardless of what the library declares.
- fast additionally swaps each thermal component's ``model`` to its
  structurally different fast counterpart (it drops the integer commitment
  variables entirely rather than relaxing them) — see ``FAST_MODEL_BY_BASE``.

This avoids writing/copying YAML files to disk for every variant; every
variant is a `.model_copy(deep=True)` of a schema parsed once from a small,
shared pool of committed fixture files (``libs/thermal_heuristic.yml`` for
model libraries, ``optim-config/thermal_heuristic.yml`` for heuristic
configs — both shared across every case and mode).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import pytest

from gems_craft.model.parsing import LibrarySchema, parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionConfig,
    ScenarioScopeConfig,
    TimeScopeConfig,
    load_optim_config,
)
from gems_craft.study.parsing import (
    HeuristicId,
    IntegerStrategy,
    IntegerStrategyId,
    SystemSchema,
    parse_yaml_system,
)
from gems_craft.study.resolve_components import (
    build_data_base,
    resolve_system,
)
from gems_craft.study.study import Study
from gems_runner.simulation.simulation_table import SimulationTable

STUDIES_DIR = Path(__file__).parent / "studies"
SHARED_LIB_FILE = Path(__file__).parent / "libs" / "thermal_heuristic.yml"
OPTIM_CONFIG_FILE = Path(__file__).parent / "optim-config" / "thermal_heuristic.yml"

# fast mode drops the integer commitment variables entirely rather than relaxing
# them, so it needs a structurally different model than milp/lp/accurate — both
# variants live side by side in the single shared library file.
FAST_MODEL_BY_BASE = {
    "antares_legacy_models.thermal": "antares_legacy_models.thermal_fast",
    "antares_legacy_models.thermal_with_ramp": "antares_legacy_models.thermal_fast_with_ramp",
}


# ---------------------------------------------------------------------------
# In-memory schema mutation
# ---------------------------------------------------------------------------


def with_integer_strategy(
    system: SystemSchema,
    component_ids: List[str],
    mode: str,
) -> SystemSchema:
    """Return a deep copy of *system* with the given components' integer-strategy set.

    ``mode`` is "lp" (-> relaxed), "accurate", or "fast" (-> heuristic with that id).
    In "fast" mode, each component's ``model`` is additionally swapped to its
    fast counterpart via ``FAST_MODEL_BY_BASE``.
    """
    strategy = (
        IntegerStrategy(id=IntegerStrategyId.RELAXED)
        if mode == "lp"
        else IntegerStrategy(
            id=IntegerStrategyId.HEURISTIC, heuristic_id=HeuristicId(mode)
        )
    )
    new_system = system.model_copy(deep=True)
    for comp in new_system.components:
        if comp.id in component_ids:
            comp.integer_strategy = strategy
            if mode == "fast":
                comp.model = FAST_MODEL_BY_BASE[comp.model]
    return new_system


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalCaseSpec:
    base_dir: str
    thermal_components: List[str]


CASES: Dict[str, ThermalCaseSpec] = {
    "one_cluster": ThermalCaseSpec(
        base_dir="thermal_heuristic_one_cluster",
        thermal_components=["G"],
    ),
    "three_clusters": ThermalCaseSpec(
        base_dir="thermal_heuristic_three_clusters",
        thermal_components=["G1", "G2", "G3"],
    ),
    "two_clusters_low_load": ThermalCaseSpec(
        base_dir="thermal_heuristic_two_clusters_low_load",
        thermal_components=["G1", "G2"],
    ),
    "two_clusters_with_bc": ThermalCaseSpec(
        base_dir="thermal_heuristic_two_clusters_with_bc",
        thermal_components=["G1", "G2"],
    ),
    "one_cluster_with_ramp": ThermalCaseSpec(
        base_dir="thermal_heuristic_one_cluster_with_ramp",
        thermal_components=["G"],
    ),
}


@lru_cache
def _base_system(base_dir: str) -> SystemSchema:
    with (STUDIES_DIR / base_dir / "input" / "system.yml").open() as f:
        return parse_yaml_system(f)


@lru_cache
def _shared_library() -> LibrarySchema:
    with SHARED_LIB_FILE.open() as f:
        return parse_yaml_library(f)


def build_thermal_study(case_id: str, mode: str) -> Study:
    """Build a Study in memory for (case_id, mode) in {milp, lp, accurate, fast}.

    No disk writes: every variant is derived from schemas parsed once from the
    case's committed base directory and the shared library file.
    """
    spec = CASES[case_id]
    system = _base_system(spec.base_dir)
    lib_dict = resolve_library([_shared_library()])

    if mode != "milp":
        system = with_integer_strategy(system, spec.thermal_components, mode)

    resolved_system = resolve_system(system, lib_dict)
    # model_dict = {}
    # for library in lib_dict.values():
    #     model_dict |= library.models
    # consistency_check(resolved_system, model_dict)

    series_dir = STUDIES_DIR / spec.base_dir / "input" / "data-series"
    database = build_data_base(system, series_dir)
    return Study(system=resolved_system, database=database)


# ---------------------------------------------------------------------------
# Optim-config: a single file (optim-config/thermal_heuristic.yml) shared
# across every case and every mode — it declares heuristics for all four
# thermal models (thermal/thermal_fast/thermal_with_ramp/thermal_fast_with_ramp)
# side by side. validate_optim_config rejects a ``models`` entry whose id isn't
# present in the resolved system, so callers must trim it down to just the
# entry matching whichever model the case's components actually resolve to.
# milp/lp never look up a heuristic, so they get a bare config instead.
# ---------------------------------------------------------------------------


@lru_cache
def _shared_optim_config() -> OptimConfig:
    config = load_optim_config(OPTIM_CONFIG_FILE)
    assert config is not None
    return config


@lru_cache
def _base_model_for(case_id: str) -> str:
    spec = CASES[case_id]
    system = _base_system(spec.base_dir)
    component_id = spec.thermal_components[0]
    return next(c.model for c in system.components if c.id == component_id)


def optim_config_for(
    case_id: str,
    mode: str,
    first_time_step: int,
    last_time_step: int,
    scenario_ids: List[int],
    resolution: Optional[ResolutionConfig] = None,
) -> OptimConfig:
    if mode in ("milp", "lp"):
        config = OptimConfig()
    else:
        base_model = _base_model_for(case_id)
        model_id = FAST_MODEL_BY_BASE[base_model] if mode == "fast" else base_model
        shared = _shared_optim_config()
        config = shared.model_copy(
            update={"models": [m for m in shared.models if m.id == model_id]}
        )

    update = {
        "time_scope": TimeScopeConfig(
            first_time_step=first_time_step, last_time_step=last_time_step
        ),
        "scenario_scope": ScenarioScopeConfig(include=scenario_ids),
    }
    if resolution is not None:
        update["resolution"] = resolution
    return config.model_copy(update=update)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def check_output(
    st: SimulationTable,
    component_id: str,
    output_id: str,
    expected_values: list,
    scenario_index: int = 0,
    abs: float = 1e-6,
) -> None:
    """Assert *expected_values* against *st*'s per-timestep output."""
    actual = cast(
        "pd.Series[Any]",
        st.component(component_id)
        .output(output_id)
        .value(scenario_index=scenario_index),
    )
    assert actual.tolist() == pytest.approx(expected_values, abs=abs)
