# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
Tests that integer_strategy is correctly reflected in the built linopy model.

- EXACT    → integer variables keep their type; continuous variables unaffected
- RELAXED  → integer variables become continuous; continuous variables unaffected
- HEURISTIC → same as RELAXED for the solver
"""

import pandas as pd
import pytest

from gems_craft.expression.expression import LowerBoundNode, literal, param, var
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model.model import model
from gems_craft.model.parameter import float_parameter
from gems_craft.model.variable import float_variable, int_variable
from gems_craft.study import DataBase, Study, System
from gems_craft.study.data import TimeSeriesData
from gems_craft.study.parsing import HeuristicId, IntegerStrategy, IntegerStrategyId
from gems_craft.study.system import Component
from gems_runner.simulation import TimeBlock, build_problem
from gems_runner.simulation.simulation_table import SimulationTableBuilder

MIXED_MODEL = model(
    id="mixed_model",
    variables=[
        int_variable("nb_units", lower_bound=literal(0), upper_bound=literal(10)),
        float_variable("generation", lower_bound=literal(0)),
    ],
)


def _build(strategies: list[IntegerStrategyId]) -> build_problem:  # type: ignore[valid-type]
    system = System("test")
    for i, strategy in enumerate(strategies):
        component = Component(
            model=MIXED_MODEL,
            id=f"c{i+1}",
            integer_strategy=IntegerStrategy(
                id=strategy,
                heuristic_id=(
                    HeuristicId.FAST
                    if strategy == IntegerStrategyId.HEURISTIC
                    else None
                ),
            ),
        )
        system.add_component(component)
    return build_problem(Study(system, DataBase()), TimeBlock(1, [0]), scenario_ids=[0])


def test_exact_strategy_keeps_integer_variable() -> None:
    problem = _build([IntegerStrategyId.EXACT, IntegerStrategyId.EXACT])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_relaxed_strategy_makes_integer_variable_continuous() -> None:
    problem = _build([IntegerStrategyId.RELAXED, IntegerStrategyId.RELAXED])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_heuristic_strategy_makes_integer_variable_continuous() -> None:
    problem = _build([IntegerStrategyId.HEURISTIC, IntegerStrategyId.HEURISTIC])
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )


def test_mixed_strategies_work() -> None:
    """Components with different strategies in the same model are supported."""
    problem = _build([IntegerStrategyId.EXACT, IntegerStrategyId.RELAXED])
    assert (
        problem.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )
    assert (
        problem.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )

    problem2 = _build([IntegerStrategyId.HEURISTIC, IntegerStrategyId.EXACT])
    assert (
        problem2.linopy_model.variables["mixed_model__generation"].data.attrs["integer"]
        is False
    )
    assert (
        problem2.linopy_model.variables["mixed_model__nb_units"].data.attrs["integer"]
        is True
    )
    assert (
        problem2.linopy_model.variables["mixed_model__nb_units__relaxed"].data.attrs[
            "integer"
        ]
        is False
    )


MODEL_WITH_TIME_DEPENDENT_BOUND = model(
    id="model_with_time_dependent_bound",
    parameters=[
        float_parameter("cap", structure=IndexingStructure(True, False)),
    ],
    variables=[
        int_variable("nb_units", lower_bound=literal(0), upper_bound=param("cap")),
    ],
)


def test_mixed_strategies_with_time_dependent_parameter_bound() -> None:
    """A variable's bound may reference a time-dependent parameter that varies
    per component (e.g. thermal's cluster_max_generation). When sibling
    components of the same model use different integer strategies, the split
    groups built by _create_variables_for_model must each see only their own
    components' parameter values, not every component sharing the model.
    """
    system = System("test")
    db = DataBase()
    caps = {"c1": 5.0, "c2": 10.0}
    for comp_id, strategy in zip(
        caps, [IntegerStrategyId.EXACT, IntegerStrategyId.RELAXED]
    ):
        system.add_component(
            Component(
                model=MODEL_WITH_TIME_DEPENDENT_BOUND,
                id=comp_id,
                integer_strategy=IntegerStrategy(id=strategy),
            )
        )
        db.add_data(comp_id, "cap", TimeSeriesData(pd.Series([caps[comp_id]])))

    problem = build_problem(Study(system, db), TimeBlock(1, [0]), scenario_ids=[0])

    exact_var = problem.get_component_variable(
        "model_with_time_dependent_bound", "nb_units", "c1"
    )
    relaxed_var = problem.get_component_variable(
        "model_with_time_dependent_bound", "nb_units", "c2"
    )
    assert exact_var is not None and exact_var.upper.sel(component="c1").item() == 5.0
    assert (
        relaxed_var is not None and relaxed_var.upper.sel(component="c2").item() == 10.0
    )


MIXED_MODEL_WITH_BOUND_OUTPUT = model(
    id="mixed_model_with_bound_output",
    variables=[
        float_variable("generation", lower_bound=literal(0), upper_bound=literal(100)),
    ],
    extra_outputs={
        "gen_lb": LowerBoundNode("generation"),
    },
    objective_contributions={
        "null_objective": (literal(0) * var("generation")).time_sum().expec()
    },
)


def test_lower_bound_extra_output_bypasses_merged_group_variable() -> None:
    """lower_bound() extra-outputs must read the real per-component linopy
    Variable, not the merged/detached _MergedGroupVariable copy built for
    models split across relaxed/exact strategy groups — otherwise a
    heuristic-style bound mutation on one component would not be visible (or
    would leak across components).
    """
    system = System("test")
    for comp_id, strategy in zip(
        ["c1", "c2"], [IntegerStrategyId.EXACT, IntegerStrategyId.RELAXED]
    ):
        system.add_component(
            Component(
                model=MIXED_MODEL_WITH_BOUND_OUTPUT,
                id=comp_id,
                integer_strategy=IntegerStrategy(id=strategy),
            )
        )

    problem = build_problem(
        Study(system, DataBase()), TimeBlock(1, [0]), scenario_ids=[0]
    )
    problem.solve(solver_name="highs")

    # Simulate what a heuristic does: mutate c2's bound directly via the real
    # per-component Variable, bypassing the merged/detached copy.
    c2_var = problem.get_component_variable(
        "mixed_model_with_bound_output", "generation", "c2"
    )
    assert c2_var is not None
    c2_var.lower.sel(component="c2")[:] = 42.0

    st = SimulationTableBuilder().build(problem)

    c2_lb = st.component("c2").output("gen_lb").value(time_index=0, scenario_index=0)
    c1_lb = st.component("c1").output("gen_lb").value(time_index=0, scenario_index=0)

    assert c2_lb == pytest.approx(42.0)
    assert c1_lb == pytest.approx(0.0)
