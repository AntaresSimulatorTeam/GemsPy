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
Exports, for each time block, one problem shared by all scenarios (MPS) and
its scenario-dependent entries (Parquet).

See the "Exporting parametric problems" section of
docs/user-guide/optimisation.md for the format.
"""

import json
from pathlib import Path
from typing import Dict

from gems_craft.optim_config.parsing import OptimConfig, ResolutionMode
from gems_craft.optim_config.validation import validate_optim_config
from gems_craft.study.study import Study
from gems_craft.study.validation import check_data_requirements
from gems_runner.parametric.builder import ParametricEntries, _ParametricProblemBuilder
from gems_runner.session.session import parallel_blocks, time_scope_block
from gems_runner.simulation.couplings import build_couplings, dump_couplings
from gems_runner.simulation.optimization import (
    DecompositionFilter,
    OptimizationProblem,
    OutOfBoundsFilter,
    _build_decomposed,
)
from gems_runner.simulation.time_block import TimeBlock

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"


def export_parametric_problems(
    study: Study, optim_config: OptimConfig, output_dir: Path
) -> Path:
    """Write, for each time block of *optim_config*, the problem of its first
    scenario and the entries that differ in the other scenarios.

    The blocks follow the resolution mode: one block covering the time scope
    in ``frontal`` mode, blocks of ``block-length`` time steps in
    ``parallel-subproblems`` mode, and a master and a subproblem covering the
    time scope in ``benders-decomposition`` mode.  ``sequential-subproblems``
    mode is rejected: each block starts from the solution of the previous one.

    Returns *output_dir*, which holds:

    - ``block_<id>/<name>.mps`` and ``block_<id>/<name>_parameters.parquet``
      for each problem of a block (``problem``, or ``master`` and
      ``subproblem`` with ``structure.txt`` in Benders mode);
    - ``manifest.json``, written last: ``format_version``, ``scenarios`` and
      the ordered ``blocks`` ids.
    """
    validate_optim_config(optim_config, study.system, study.scenario_builder)
    mode = optim_config.resolution.mode
    if mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
        raise ValueError(
            f"Mode '{mode.value}' cannot be exported: each block starts from "
            "the solution of the previous one. Use "
            f"'{ResolutionMode.PARALLEL_SUBPROBLEMS.value}' instead."
        )
    check_data_requirements(study)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Removed first: a folder without manifest is incomplete.
    (output_dir / MANIFEST_FILE).unlink(missing_ok=True)

    scenario_ids = optim_config.scenario_scope.scenario_ids
    if mode == ResolutionMode.BENDERS_DECOMPOSITION:
        time_blocks = [time_scope_block(optim_config, block_id=1)]
        _export_benders_block(study, optim_config, time_blocks[0], output_dir)
    else:
        time_blocks = (
            [time_scope_block(optim_config, block_id=0)]
            if mode == ResolutionMode.FRONTAL
            else parallel_blocks(optim_config)
        )
        oob_filter = OutOfBoundsFilter(optim_config)
        for block in time_blocks:
            builder = _ParametricProblemBuilder(
                name="problem",
                study=study,
                block=block,
                scenario_ids=scenario_ids,
                oob_filter=oob_filter,
            )
            _write_problem(builder.build(), builder.entries, output_dir, block)

    manifest = {
        "format_version": FORMAT_VERSION,
        "scenarios": scenario_ids,
        "blocks": [block.id for block in time_blocks],
    }
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return output_dir


def _export_benders_block(
    study: Study, optim_config: OptimConfig, block: TimeBlock, output_dir: Path
) -> None:
    oob_filter = OutOfBoundsFilter(optim_config)
    entries: Dict[str, ParametricEntries] = {}

    def build(name: str, location_filter: DecompositionFilter) -> OptimizationProblem:
        builder = _ParametricProblemBuilder(
            name=name,
            study=study,
            block=block,
            scenario_ids=optim_config.scenario_scope.scenario_ids,
            location_filter=location_filter,
            oob_filter=oob_filter,
        )
        problem = builder.build()
        entries[name] = builder.entries
        return problem

    decomposed = _build_decomposed(
        build, optim_config, subproblem_name="subproblem", master_name="master"
    )
    if decomposed.master is None:
        raise ValueError(
            "Benders decomposition requires a master problem: no element of "
            "the optim-config is located in the master."
        )
    for problem in (decomposed.master, decomposed.subproblem):
        _write_problem(problem, entries[problem.name], output_dir, block)
    dump_couplings(
        build_couplings(decomposed, optim_config), output_dir / _block_folder(block)
    )


def _block_folder(block: TimeBlock) -> str:
    return f"block_{block.id}"


def _write_problem(
    problem: OptimizationProblem,
    entries: ParametricEntries,
    output_dir: Path,
    block: TimeBlock,
) -> None:
    """Write ``<name>.mps`` and ``<name>_parameters.parquet`` in the block
    folder."""
    folder = output_dir / _block_folder(block)
    folder.mkdir(parents=True, exist_ok=True)
    problem.linopy_model.to_file(
        folder / f"{problem.name}.mps", explicit_coordinate_names=True
    )
    assert problem.scenario_ids is not None
    entries.to_frame(problem.linopy_model, problem.scenario_ids).to_parquet(
        folder / f"{problem.name}_parameters.parquet", index=False
    )
