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

"""
Functional tests for component-dependent time shift in expressions.

All tests use a ``lagged-storage`` model whose level equation references
the state ``lag`` steps ago under CYCLE border management:

    level[t] − level[t − lag] + withdrawal[t] = inflows

where ``lag`` is a **per-component constant parameter**.  When two
component instances carry different ``lag`` values, the problem builder
must emit different shifted sub-expressions for each component — the
"per-component slow path" in ``linearize._time_shift``.

Group-partition property (the key insight)
------------------------------------------
With CYCLE over H steps and a given lag k, summing the level equation
over each orbit of the cyclic permutation σ: t ↦ t − k (mod H) gives:

    sum_{t ∈ orbit} withdrawal[t] = |orbit| × inflows

where |orbit| = H / gcd(H, k).  Timesteps in *different* orbits are
completely decoupled.

Consequences tested below
--------------------------
* lag=1, H=4: gcd=1, one orbit of size 4.  All four steps are linked;
  withdrawal can be freely redistributed across them.

* lag=2, H=4: gcd=2, two orbits {0,2} and {1,3} each of size 2.
  Even and odd steps are decoupled: each group is forced to withdraw
  exactly 2 × inflows in total.  If demand is concentrated at even
  steps, the odd group produces unavoidable spillage.

* lag=3, H=6: gcd=3, three orbits {0,3}, {1,4}, {2,5} each of size 2.
  Each pair sums to 2 × inflows.  If demand falls only on {0,3}, the
  other four steps generate forced spillage.

Test scenarios
--------------
1. **Single component, lag=2, H=4** (scalar DataArray path):
     demand=[5, 0, 5, 0] — exactly matches even-group supply (10).
     Odd group must discharge 10 units where demand=0 → spillage=10.
     Objective = 10 × spillage_cost(1) = 10.

2. **Two components, lag=1 and lag=2, H=4** (per-component slow path):
     demand=[20, 0, 20, 0].
     lag=1 contributes up to 20 to even demand steps (10+10).
     lag=2 even-group contributes 10 to even steps; odd-group forces
     spillage=10.
     Total even supply = 30 < 40 → unsupplied=10.
     Objective = 10×100 + 10×1 = 1010.

3. **Three components, lag=1,2,3, H=6** (slow path, three unique shifts):
     demand=[30, 0, 0, 30, 0, 0].
     lag=1: freely provides 30 at demand steps.
     lag=2: even/odd split aligns perfectly with demand at t=0/t=3.
     lag=3: orbits {0,3},{1,4},{2,5}; demand-orbit sum=10, but demand
     already covered → 10 excess spillage there; 20 forced spillage
     from the other two orbits.
     Total spillage = 30. Objective = 30×1 = 30.

4. **YAML-based version of scenario 2** (end-to-end YAML path):
     Same two-component setup from files; expected objective = 1010.
"""

from pathlib import Path

import pandas as pd
import pytest

from gems.expression import literal, param, var
from gems.expression.indexing_structure import IndexingStructure
from gems.model import ModelPort, float_parameter, float_variable, model
from gems.model.constraint import Constraint
from gems.model.parsing import parse_yaml_library
from gems.model.port import PortFieldDefinition, PortFieldId
from gems.model.resolve_library import resolve_library
from gems.simulation import TimeBlock, build_problem
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    Study,
    System,
    TimeScenarioSeriesData,
    create_component,
)
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from tests.e2e.functional.libs.standard import (
    BALANCE_PORT_TYPE,
    DEMAND_MODEL,
    NODE_BALANCE_MODEL,
    SPILLAGE_MODEL,
    UNSUPPLIED_ENERGY_MODEL,
)

CONSTANT = IndexingStructure(False, False)

# ---------------------------------------------------------------------------
# Shared model definition
# ---------------------------------------------------------------------------

# Storage whose level equation uses a *parametrised* lag.
# Constraint:  level[t] - level[t - lag] + withdrawal[t] = inflows
# Port:        flow = withdrawal  (positive → supplies the network)
LAGGED_STORAGE_MODEL = model(
    id="LAGGED_STORAGE",
    parameters=[
        float_parameter("lag", CONSTANT),
        float_parameter("p_max", CONSTANT),
        float_parameter("level_max", CONSTANT),
        float_parameter("inflows", CONSTANT),
    ],
    variables=[
        float_variable("level", lower_bound=literal(0), upper_bound=param("level_max")),
        float_variable(
            "withdrawal", lower_bound=literal(0), upper_bound=param("p_max")
        ),
    ],
    ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
    port_fields_definitions=[
        PortFieldDefinition(
            port_field=PortFieldId("balance_port", "flow"),
            definition=var("withdrawal"),
        )
    ],
    constraints=[
        Constraint(
            name="Level equation",
            expression=(
                var("level") - var("level").shift(-param("lag")) + var("withdrawal")
                == param("inflows")
            ),
        )
    ],
)

SPILLAGE_COST = 1.0
UNSUPPLIED_COST = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demand_series(values: list[float]) -> TimeScenarioSeriesData:
    """Build a single-scenario time-series DataArray from a list of values."""
    df = pd.DataFrame(values, columns=[0])
    return TimeScenarioSeriesData(df)


def _base_database(
    demand_data: TimeScenarioSeriesData | ConstantData,
) -> DataBase:
    database = DataBase()
    database.add_data("D", "demand", demand_data)
    database.add_data("S", "cost", ConstantData(SPILLAGE_COST))
    database.add_data("U", "cost", ConstantData(UNSUPPLIED_COST))
    return database


def _add_storage(
    database: DataBase,
    component_id: str,
    *,
    lag: int,
    inflows: float,
    p_max: float,
    level_max: float = 1000.0,
) -> None:
    database.add_data(component_id, "lag", ConstantData(lag))
    database.add_data(component_id, "p_max", ConstantData(p_max))
    database.add_data(component_id, "level_max", ConstantData(level_max))
    database.add_data(component_id, "inflows", ConstantData(inflows))


def _build_system(*storage_ids: str) -> System:
    node = Component(model=NODE_BALANCE_MODEL, id="N")
    demand_comp = create_component(model=DEMAND_MODEL, id="D")
    spillage_comp = create_component(model=SPILLAGE_MODEL, id="S")
    unsupplied_comp = create_component(model=UNSUPPLIED_ENERGY_MODEL, id="U")
    storage_comps = [
        create_component(model=LAGGED_STORAGE_MODEL, id=sid) for sid in storage_ids
    ]

    system = System("test")
    system.add_component(node)
    for comp in [demand_comp, spillage_comp, unsupplied_comp] + storage_comps:
        system.add_component(comp)
        system.connect(PortRef(comp, "balance_port"), PortRef(node, "balance_port"))
    return system


# ---------------------------------------------------------------------------
# Test 1 — single component, lag=2, H=4  (scalar DataArray path)
# ---------------------------------------------------------------------------


def test_single_lag2_forced_odd_spillage() -> None:
    """
    Single lagged-storage with lag=2 over H=4 time steps.

    Demand = [5, 0, 5, 0] — concentrated at even time steps (t=0, t=2).

    Group-partition (lag=2, H=4, gcd=2):
      even orbit {0, 2}: withdrawal sums to 2 × inflows = 10 → exactly covers
                         even demand (5 + 5 = 10).  No unsupplied energy.
      odd orbit  {1, 3}: withdrawal sums to 10 but demand = 0 there →
                         spillage = 10 (unavoidable regardless of odd split).

    With spillage_cost=1 and unsupplied_cost=100:
      objective = 10 × 1 = 10.

    The lag parameter is a scalar DataArray (one component) — exercises the
    scalar code path in ``_time_shift``, not the per-component slow path.
    """
    horizon = 4
    inflows = 5.0

    database = _base_database(_make_demand_series([5.0, 0.0, 5.0, 0.0]))
    _add_storage(database, "STS", lag=2, inflows=inflows, p_max=10.0)

    system = _build_system("STS")
    problem = build_problem(
        Study(system, database),
        TimeBlock(1, list(range(horizon))),
        scenario_ids=list(range(1)),
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    # Odd orbit forces 10 units of spillage; no unsupplied energy.
    assert problem.objective_value == pytest.approx(10.0 * SPILLAGE_COST)


# ---------------------------------------------------------------------------
# Test 2 — two components, lag=1 and lag=2, H=4  (per-component slow path)
# ---------------------------------------------------------------------------


def test_two_components_different_lags_asymmetric_demand() -> None:
    """
    Two lagged-storage instances: STS_lag1 (lag=1) and STS_lag2 (lag=2).
    Both have inflows=5/step, p_max=10, over H=4 steps.

    Demand = [20, 0, 20, 0] — all demand at even time steps.

    Analysis:
      STS_lag1 (lag=1, gcd=1, one orbit of size 4):
        The storage can freely concentrate its 20 forced units at even
        steps → contributes 10 at t=0 and 10 at t=2.
        No spillage from STS_lag1.

      STS_lag2 (lag=2, gcd=2, two orbits of size 2):
        Even orbit {0,2}: sum = 10  → at most 10 units to even demand steps.
        Odd  orbit {1,3}: sum = 10  → 10 units of forced spillage (demand=0).

    Balance:
      Even-step supply = 20 (lag1) + 10 (lag2) = 30 < 40 (demand) → unsupplied = 10.
      Odd-step spillage = 10.

    With spillage_cost=1 and unsupplied_cost=100:
      objective = 10 × 100 + 10 × 1 = 1010.

    The ``lag`` parameter DataArray has a component dimension with two
    distinct values ([−1, −2] after negation), triggering the per-component
    slow path in ``linearize._time_shift``.
    """
    horizon = 4
    inflows = 5.0

    database = _base_database(_make_demand_series([20.0, 0.0, 20.0, 0.0]))
    _add_storage(database, "STS1", lag=1, inflows=inflows, p_max=10.0)
    _add_storage(database, "STS2", lag=2, inflows=inflows, p_max=10.0)

    system = _build_system("STS1", "STS2")
    problem = build_problem(
        Study(system, database),
        TimeBlock(1, list(range(horizon))),
        scenario_ids=list(range(1)),
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    # 10 unsupplied (even demand not covered) + 10 spillage (odd forced).
    expected = 10.0 * UNSUPPLIED_COST + 10.0 * SPILLAGE_COST
    assert problem.objective_value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3 — three components, lag=1,2,3, H=6  (slow path, three unique shifts)
# ---------------------------------------------------------------------------


def test_three_components_distinct_lags_orbit_spillage() -> None:
    """
    Three lagged-storage instances: lag=1, lag=2, lag=3 over H=6 steps.
    All have inflows=5/step, p_max=15.

    Demand = [30, 0, 0, 30, 0, 0] — at t=0 and t=3 only.

    Group-partition analysis:
      lag=1, gcd(6,1)=1 → one orbit of size 6.
        Freely provides 30 at demand steps (15 at t=0, 15 at t=3). ✓

      lag=2, gcd(6,2)=2 → even orbit {0,2,4} sum=15, odd orbit {1,3,5} sum=15.
        Even orbit: up to 15 at t=0.  Odd orbit: up to 15 at t=3.
        Provides 30 at demand steps with no forced off-demand production. ✓

      lag=3, gcd(6,3)=3 → orbits {0,3}, {1,4}, {2,5} each sum=10.
        Orbit {0,3}: forced total = 10 ≡ excess at demand steps (already covered).
        Orbits {1,4} and {2,5}: forced total = 10 + 10 = 20 at non-demand steps
        → 20 units of unavoidable spillage.
        The 10 units from orbit {0,3} are also excess (demand covered by lag1+lag2)
        → 10 more units of spillage.

    Balance:
      Demand steps covered: lag1(30) + lag2(30) = 60 = demand(60). ✓
      lag3 adds 10 excess at demand steps → spillage = 10 there.
      lag3 forces 20 spillage at non-demand steps.
      Total spillage = 30. Unsupplied = 0.

    With spillage_cost=1:
      objective = 30 × 1 = 30.

    The ``lag`` parameter DataArray carries three distinct values, so the
    slow-path masking loop runs three iterations.
    """
    horizon = 6
    inflows = 5.0

    database = _base_database(_make_demand_series([30.0, 0.0, 0.0, 30.0, 0.0, 0.0]))
    for lag, sid in [(1, "STS1"), (2, "STS2"), (3, "STS3")]:
        _add_storage(database, sid, lag=lag, inflows=inflows, p_max=15.0)

    system = _build_system("STS1", "STS2", "STS3")
    problem = build_problem(
        Study(system, database),
        TimeBlock(1, list(range(horizon))),
        scenario_ids=list(range(1)),
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    # lag=3 forces 30 total spillage; no unsupplied.
    assert problem.objective_value == pytest.approx(30.0 * SPILLAGE_COST)


# ---------------------------------------------------------------------------
# Test 4 — YAML-based end-to-end: two components, lag=1 and lag=2, H=4
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _libs_dir() -> Path:
    return Path(__file__).parent / "libs"


@pytest.fixture(scope="module")
def _systems_dir() -> Path:
    return Path(__file__).parent / "systems"


@pytest.fixture(scope="module")
def _series_dir() -> Path:
    return Path(__file__).parent / "series"


def test_two_components_different_lags_yaml(
    _libs_dir: Path, _systems_dir: Path, _series_dir: Path
) -> None:
    """
    YAML-based end-to-end counterpart of
    ``test_two_components_different_lags_asymmetric_demand``.

    Reads the ``time_shift_test`` library from ``lib_time_shift.yml`` and
    the system from ``system_time_shift_per_component.yml``.
    Demand is loaded from the ``demand_shift_test`` time series file
    ([20, 0, 20, 0]) from the series directory.

    Same two-component setup (STS_lag1: lag=1, STS_lag2: lag=2, both
    inflows=5, p_max=10) over H=4 steps with CYCLE.

    Expected: same forced unsupplied (10) and spillage (10) as the
    Python test → objective = 10 × 100 + 10 × 1 = 1010.
    """
    lib_file = _libs_dir / "lib_time_shift.yml"
    system_file = _systems_dir / "system_time_shift_per_component.yml"

    with lib_file.open() as f:
        input_library = parse_yaml_library(f)
    with system_file.open() as f:
        input_system = parse_yaml_components(f)

    lib_dict = resolve_library([input_library])
    system = resolve_system(input_system, lib_dict)
    consistency_check(system, lib_dict["time_shift_test"].models)

    database = build_data_base(input_system, timeseries_dir=_series_dir)

    problem = build_problem(
        Study(system, database),
        TimeBlock(1, list(range(4))),
        scenario_ids=list(range(1)),
    )
    problem.solve(solver_name="highs")

    assert problem.termination_condition == "optimal"
    expected = 10.0 * UNSUPPLIED_COST + 10.0 * SPILLAGE_COST
    assert problem.objective_value == pytest.approx(expected)
