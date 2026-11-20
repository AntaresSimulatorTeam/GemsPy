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
Exports one problem shared by all time blocks and scenarios (MPS) and its
entries that depend on the block or the scenario (Parquet).

See the "Exporting parametric problems" section of
docs/user-guide/optimisation.md for the format.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from gems_craft.optim_config.parsing import OptimConfig, ResolutionMode
from gems_craft.optim_config.validation import (
    check_variables_depend_on_scenario,
    validate_optim_config,
)
from gems_craft.study.study import Study
from gems_craft.study.validation import check_data_requirements
from gems_runner.parametric.builder import _ParametricProblemBuilder
from gems_runner.simulation.couplings import build_couplings, dump_couplings
from gems_runner.simulation.optimization import (
    DecompositionFilter,
    OptimizationProblem,
    OutOfBoundsFilter,
    _build_decomposed,
    _has_any_master_element,
)
from gems_runner.simulation.time_block import parallel_blocks, time_scope_block

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"


def export_parametric_problems(
    study: Study, optim_config: OptimConfig, output_dir: Path
) -> Path:
    """Write one problem shared by all time blocks and scenarios of
    *optim_config* (that of the first block and scenario), and its entries
    that differ in the other blocks and scenarios.

    The blocks follow the resolution mode: one block covering the time scope
    in ``frontal`` and ``benders-decomposition`` modes (a master and a
    subproblem in the latter), blocks of ``block-length`` time steps in
    ``parallel-subproblems`` mode, which must all have the same length.
    ``sequential-subproblems`` mode is rejected: each block starts from the
    solution of the previous one.

    Each block and scenario is exported as a problem of its own.  With
    several scenarios, a variable that does not depend on the scenario is
    therefore rejected, except in the Benders subproblem when it is located
    in ``master-and-subproblems`` (fixed by the master).

    Returns *output_dir*, which holds:

    - ``<name>.mps`` and ``<name>_parameters.parquet`` for each problem
      (``problem``, or ``master`` and ``subproblem`` with ``structure.txt`` in
      Benders mode);
    - ``manifest.json``, written last: ``format_version``, the names of the
      ``problems``, ``scenarios`` and the ordered ``blocks`` ids.
    """
    validate_optim_config(optim_config, study.system, study.scenario_builder)
    mode = optim_config.resolution.mode
    if mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
        raise ValueError(
            f"Mode '{mode.value}' cannot be exported: each block starts from "
            "the solution of the previous one. Use "
            f"'{ResolutionMode.PARALLEL_SUBPROBLEMS.value}' instead."
        )
    if mode == ResolutionMode.PARALLEL_SUBPROBLEMS:
        blocks = parallel_blocks(optim_config)
        if len(blocks[-1].timesteps) != len(blocks[0].timesteps):
            raise ValueError(
                "All time blocks must have the same length: the number of time "
                "steps of the time scope must be a multiple of 'block-length'."
            )
    else:
        block_id = 1 if mode == ResolutionMode.BENDERS_DECOMPOSITION else 0
        blocks = [time_scope_block(optim_config, block_id=block_id)]
    if mode == ResolutionMode.BENDERS_DECOMPOSITION and not _has_any_master_element(
        optim_config
    ):
        raise ValueError(
            "Benders decomposition requires a master problem: no element of "
            "the optim-config is located in the master."
        )
    if mode != ResolutionMode.PARALLEL_SUBPROBLEMS:  # parallel: checked above
        check_variables_depend_on_scenario(optim_config, study.system)
    check_data_requirements(study)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Removed first: a folder without manifest is incomplete.
    (output_dir / MANIFEST_FILE).unlink(missing_ok=True)

    scenario_ids = optim_config.scenario_scope.scenario_ids
    oob_filter = OutOfBoundsFilter(optim_config)
    builders: Dict[str, _ParametricProblemBuilder] = {}

    def build(
        name: str, location_filter: Optional[DecompositionFilter] = None
    ) -> OptimizationProblem:
        builder = _ParametricProblemBuilder(
            name=name,
            study=study,
            blocks=blocks,
            scenario_ids=scenario_ids,
            location_filter=location_filter,
            oob_filter=oob_filter,
        )
        builders[name] = builder
        return builder.build()

    if mode == ResolutionMode.BENDERS_DECOMPOSITION:
        decomposed = _build_decomposed(
            build, optim_config, subproblem_name="subproblem", master_name="master"
        )
        assert decomposed.master is not None  # checked above
        problems = [decomposed.master, decomposed.subproblem]
        dump_couplings(build_couplings(decomposed, optim_config), output_dir)
    else:
        problems = [build("problem")]

    for problem in problems:
        _write_problem(problem, builders[problem.name], output_dir)

    manifest = {
        "format_version": FORMAT_VERSION,
        "problems": [problem.name for problem in problems],
        "scenarios": scenario_ids,
        "blocks": [block.id for block in blocks],
    }
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return output_dir


def _write_problem(
    problem: OptimizationProblem, builder: _ParametricProblemBuilder, output_dir: Path
) -> None:
    """Write ``<name>.mps`` and ``<name>_parameters.parquet`` in *output_dir*."""
    problem.linopy_model.to_file(
        output_dir / f"{problem.name}.mps", explicit_coordinate_names=True
    )
    builder.entries.to_frame(problem.linopy_model, builder.realizations).to_parquet(
        output_dir / f"{problem.name}_parameters.parquet", index=False
    )
