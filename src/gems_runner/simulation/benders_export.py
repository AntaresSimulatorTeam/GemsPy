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

"""Writes the on-disk inputs (MPS files, structure.txt, options.json) needed
to invoke the external AntaresXpansion ``benders`` binary."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gems_runner.simulation.couplings import build_couplings, dump_couplings

if TYPE_CHECKING:
    from gems_craft.optim_config.parsing import OptimConfig
    from gems_runner.simulation.optimization import DecomposedProblems


def export_benders_problem(
    decomposed: "DecomposedProblems",
    optim_config: "OptimConfig",
    output_dir: Path,
    *,
    n_scenarios: int,
) -> None:
    """Write master.mps, one ``<subproblem.name>.mps`` per subproblem,
    structure.txt, and options.json into *output_dir*.

    *output_dir* becomes the ``benders`` binary's working directory
    (``AntaresXpansionCommandRunner`` chdir's into ``emplacement``), so every
    path in ``options.json`` is relative to it.

    *n_scenarios* drives the uniform ``SLAVE_WEIGHT_VALUE`` divisor: every
    subproblem (regardless of which block it covers) gets the same constant
    weight ``1 / n_scenarios``. This is only correct because within one
    Monte Carlo year, block-subproblems partition that year's cost and must
    sum unweighted, while across years the master needs the *expected*
    (probability-weighted) annual cost. ``ScenarioScopeConfig`` carries no
    per-scenario probability today, so uniform weighting is the only
    defensible default — revisit if that changes.
    """
    if decomposed.master is None:
        raise ValueError("export_benders_problem requires a master problem")

    output_dir.mkdir(parents=True, exist_ok=True)
    # JSON_FILE/LAST_ITERATION_JSON_FILE must point to a path whose parent
    # directory already exists — the binary does not create it — so give
    # them their own subdirectory (mirroring AntaresXpansion's own
    # "expansion/out.json" convention) rather than leaving them at the
    # default "." (a directory, not a writable file path).
    (output_dir / "expansion").mkdir(parents=True, exist_ok=True)

    decomposed.master.linopy_model.to_file(output_dir / f"{decomposed.master.name}.mps")
    for subproblem in decomposed.subproblems:
        subproblem.linopy_model.to_file(output_dir / f"{subproblem.name}.mps")

    dump_couplings(build_couplings(decomposed, optim_config), output_dir)

    options = {
        "MASTER_NAME": decomposed.master.name,
        "STRUCTURE_FILE": "structure.txt",
        "INPUTROOT": ".",
        "SLAVE_WEIGHT": "CONSTANT",
        "SLAVE_WEIGHT_VALUE": n_scenarios,
        # Open-source, license-free default; GemsPy's own
        # optim_config.solver_options.name (e.g. "highs") selects the solver
        # for GemsPy's own linopy solves and is not a valid AntaresXpansion
        # value ("XPRESS"/"COIN" are the only documented options) — do not
        # thread it through here.
        "SOLVER_NAME": "COIN",
        "JSON_FILE": "expansion/out.json",
        "LAST_ITERATION_JSON_FILE": "expansion/last_iteration.json",
    }
    (output_dir / "options.json").write_text(json.dumps(options, indent=5) + "\n")
