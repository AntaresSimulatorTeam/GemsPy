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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from gems.optim_config.parsing import OptimConfig
    from gems.simulation.optimization import DecomposedProblems


@dataclass(frozen=True)
class CouplingRow:
    """A single entry in the Benders structure file."""

    problem_id: str
    component_id: str
    variable_int_id: int


def _format_row(row: CouplingRow) -> str:
    return f"{row.problem_id:>24}{row.component_id:>48}{row.variable_int_id:>9}"


def _master_coupling_row(
    decomposed: "DecomposedProblems",
    model_id: str,
    var_id: str,
    comp_id: str,
) -> Optional[CouplingRow]:
    if decomposed.master is None:
        return None
    labels = decomposed.master.get_variable_labels(model_id, var_id)
    if labels is None:
        return None
    return CouplingRow(
        problem_id=decomposed.master.name,
        component_id=comp_id,
        variable_int_id=int(labels.sel(component=comp_id).item()),
    )


def _subproblem_coupling_row(
    decomposed: "DecomposedProblems",
    model_id: str,
    var_id: str,
    comp_id: str,
) -> Optional[CouplingRow]:
    labels = decomposed.subproblem.get_variable_labels(model_id, var_id)
    if labels is None:
        return None
    return CouplingRow(
        problem_id=decomposed.subproblem.name,
        component_id=comp_id,
        variable_int_id=int(labels.sel(component=comp_id).item()),
    )


def _coupling_rows_for_variable(
    decomposed: "DecomposedProblems",
    model_id: str,
    var_id: str,
) -> List[CouplingRow]:
    rows: List[CouplingRow] = []
    for comp in decomposed.subproblem.study.model_components.get(model_id, []):
        for row in (
            _master_coupling_row(decomposed, model_id, var_id, comp.id),
            _subproblem_coupling_row(decomposed, model_id, var_id, comp.id),
        ):
            if row is not None:
                rows.append(row)
    return rows


def build_couplings(
    decomposed: "DecomposedProblems",
    optim_config: "OptimConfig",
) -> List[CouplingRow]:
    from gems.optim_config.parsing import ElementLocation

    rows: List[CouplingRow] = []
    for mc in optim_config.models:
        if mc.model_decomposition is not None:
            for var_cfg in mc.model_decomposition.variables:
                if var_cfg.location == ElementLocation.MASTER_AND_SUBPROBLEMS:
                    rows.extend(
                        _coupling_rows_for_variable(decomposed, mc.id, var_cfg.id)
                    )
    return rows


def dump_couplings(rows: List[CouplingRow], output_dir: Path) -> None:
    """Write ``structure.txt`` to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure.txt").write_text(
        "\n".join(_format_row(row) for row in rows) + "\n"
    )
