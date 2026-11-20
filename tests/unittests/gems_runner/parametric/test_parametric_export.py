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
`export_parametric_problems` writes one problem for all blocks and scenarios,
and the entries that depend on them.  Each (block, scenario) rebuilt from the
export must have the objective of the block built for that scenario alone.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from gems_craft.expression.expression import (
    ExpressionNode,
    literal,
    maximum,
    param,
    var,
)
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model import Constraint, float_parameter, float_variable, model
from gems_craft.optim_config.parsing import OptimConfig
from gems_craft.study import (
    ConstantData,
    DataBase,
    ScenarioSeriesData,
    Study,
    System,
    TimeScenarioSeriesData,
    TimeSeriesData,
    create_component,
)
from gems_runner.parametric import export_parametric_problems
from gems_runner.session.session import parallel_blocks
from gems_runner.simulation import TimeBlock, build_problem
from tests.unittests.gems_runner.parametric.reconstruct import solve_realization

CONSTANT = IndexingStructure(False, False)
SCENARIO_ONLY = IndexingStructure(False, True)
TIME_ONLY = IndexingStructure(True, False)
TIME_SCENARIO = IndexingStructure(True, True)

SCENARIOS = [0, 1, 2]
TIME_STEPS = 2


def _study(
    gen_upper: ExpressionNode = literal(50),
    availability: ExpressionNode = literal(1),
    gen_cost: ExpressionNode = param("cost"),
    p_values: Optional[List[float]] = None,
) -> Study:
    """One plant (`gen` bounded by `availability * cap`) meeting a demand,
    with unsupplied energy at 1000 per unit."""
    plant = model(
        id="PLANT",
        parameters=[
            float_parameter("demand", TIME_SCENARIO),
            float_parameter("export", TIME_ONLY),
            float_parameter("p", SCENARIO_ONLY),
            float_parameter("q", CONSTANT),
            float_parameter("cost", CONSTANT),
        ],
        variables=[
            float_variable("gen", lower_bound=literal(0), upper_bound=gen_upper),
            float_variable("unsupplied", lower_bound=literal(0)),
            float_variable(
                "cap",
                lower_bound=literal(0),
                upper_bound=literal(100),
                structure=CONSTANT,
            ),
        ],
        constraints=[
            Constraint(
                "balance",
                var("gen") + var("unsupplied") == param("demand") + param("export"),
            ),
            Constraint("availability", var("gen") <= availability * var("cap")),
        ],
        objective_contributions={
            "operation": (gen_cost * var("gen") + 1000 * var("unsupplied"))
            .time_sum()
            .expec(),
            "investment": 3 * var("cap"),
        },
    )
    database = DataBase()
    database.add_data(
        "P",
        "demand",
        TimeScenarioSeriesData(pd.DataFrame([[5, 7, 9], [6, 8, 4]], dtype=float)),
    )
    database.add_data(
        "P", "p", ScenarioSeriesData(np.array(p_values or [0.5, 0.8, 1.0]))
    )
    database.add_data("P", "export", TimeSeriesData(pd.Series([1.0, 3.0])))
    database.add_data("P", "q", ConstantData(0.7))
    database.add_data("P", "cost", ConstantData(2))
    system = System("parametric")
    system.add_component(create_component(model=plant, id="P"))
    return Study(system, database)


def _config(mode: str = "frontal", block_length: Optional[int] = None) -> OptimConfig:
    resolution: Dict[str, object] = {"mode": mode}
    if block_length is not None:
        resolution["block-length"] = block_length
    return OptimConfig.model_validate(
        {
            "time-scope": {"first-time-step": 0, "last-time-step": TIME_STEPS - 1},
            "scenario-scope": {"include": [f"0-{len(SCENARIOS) - 1}"]},
            "resolution": resolution,
        }
    )


def _parameters(output_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(output_dir / "problem_parameters.parquet")


def _assert_realizations_match(
    study: Study, output_dir: Path, blocks: List[TimeBlock]
) -> None:
    """Each (block, scenario) rebuilt from the export has the objective of the
    block built for that scenario alone."""
    parameters = _parameters(output_dir)
    for block in blocks:
        for s in SCENARIOS:
            reference = build_problem(study, block, [s])
            reference.solve(solver_name="highs", output_flag=False)
            assert solve_realization(
                output_dir / "problem.mps", parameters, block.id, s
            ) == pytest.approx(reference.objective_value)


@pytest.mark.parametrize(
    ("study_kwargs", "kinds"),
    [
        pytest.param({}, {"rhs"}, id="rhs"),
        pytest.param({"availability": param("p")}, {"rhs", "coef"}, id="coefficient"),
        pytest.param({"gen_upper": 10 * param("p")}, {"rhs", "upper"}, id="bound"),
        pytest.param({"gen_cost": 4 * param("p")}, {"rhs", "obj"}, id="objective"),
        pytest.param(
            {"availability": maximum(param("p"), param("q"))},
            {"rhs", "coef"},
            id="nonlinear-in-parameters",
        ),
    ],
)
def test_each_scenario_rebuilt_from_the_export_has_its_own_objective(
    study_kwargs: Dict[str, ExpressionNode], kinds: set, tmp_path: Path
) -> None:
    study = _study(**study_kwargs)
    export_parametric_problems(study, _config(), tmp_path)

    assert set(_parameters(tmp_path)["kind"]) == kinds
    _assert_realizations_match(study, tmp_path, [TimeBlock(0, list(range(TIME_STEPS)))])


def test_only_entries_that_differ_between_scenarios_are_written(
    tmp_path: Path,
) -> None:
    """A scenario-dependent parameter with the same value in every scenario
    writes no entry: the nominal problem already holds it."""
    export_parametric_problems(
        _study(availability=param("p"), p_values=[0.8, 0.8, 0.8]), _config(), tmp_path
    )
    parameters = _parameters(tmp_path)

    assert set(parameters["kind"]) == {"rhs"}
    # The demand of time step 0 differs in every scenario, that of 1 too.
    assert parameters.groupby("row").size().tolist() == [3, 3]


def test_parallel_mode_exports_one_problem_for_all_blocks(tmp_path: Path) -> None:
    """The time-dependent parameters (`demand`, and `export` which does not
    depend on the scenario) vary between blocks: their entries are in the
    Parquet file, per block and scenario."""
    config = _config("parallel-subproblems", block_length=1)
    study = _study()
    export_parametric_problems(study, config, tmp_path)

    blocks = parallel_blocks(config)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {
        "format_version": 1,
        "scenarios": SCENARIOS,
        "blocks": [b.id for b in blocks],
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "manifest.json",
        "problem.mps",
        "problem_parameters.parquet",
    ]
    _assert_realizations_match(study, tmp_path, blocks)


def test_blocks_of_different_lengths_are_rejected_before_writing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "export"
    config = OptimConfig.model_validate(
        {
            "time-scope": {"first-time-step": 0, "last-time-step": 2},
            "resolution": {"mode": "parallel-subproblems", "block-length": 2},
        }
    )
    with pytest.raises(ValueError, match="multiple of 'block-length'"):
        export_parametric_problems(_study(), config, output_dir)
    assert not output_dir.exists()


def test_sequential_mode_is_rejected_before_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "export"
    config = OptimConfig.model_validate(
        {
            "time-scope": {"first-time-step": 0, "last-time-step": 1},
            "resolution": {"mode": "sequential-subproblems", "block-length": 1},
        }
    )
    with pytest.raises(ValueError, match="parallel-subproblems"):
        export_parametric_problems(_study(), config, output_dir)
    assert not output_dir.exists()
