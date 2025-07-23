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
The xpansion module extends the optimization module
with Benders solver related functions
"""

import pathlib
from typing import Any, Dict, List, Optional

from gems.simulation.decision_tree import DecisionTreeNode
from gems.simulation.optimization import (
    BlockBorderManagement,
    OptimizationProblem,
    build_problem,
    fusion_problems,
)
from gems.simulation.output_values import (
    BendersDecomposedSolution,
    BendersMergedSolution,
    BendersSolution,
)
from gems.simulation.runner import BendersRunner, MergeMPSRunner
from gems.simulation.strategy import (
    ExpectedValue,
    InvestmentProblemStrategy,
    OperationalProblemStrategy,
)
from gems.simulation.time_block import TimeBlock
from gems.study.data import DataBase
from gems.utils import read_json, serialize, serialize_json


class BendersDecomposedProblem:
    """
    A simpler interface for the Benders Decomposed problem
    """

    master: OptimizationProblem
    subproblems: List[OptimizationProblem]

    emplacement: pathlib.Path
    output_path: pathlib.Path
    structure_filename: str

    solution: Optional[BendersSolution]
    is_merged: bool

    def __init__(
        self,
        master: OptimizationProblem,
        subproblems: List[OptimizationProblem],
        emplacement: str = "outputs/lp",
        output_path: str = "expansion",
        struct_filename: str = "structure.txt",
    ) -> None:
        self.master = master
        self.subproblems = subproblems

        self.emplacement = pathlib.Path(emplacement)
        self.output_path = pathlib.Path(output_path)
        self.structure_filename = struct_filename

        self.solution = None
        self.is_merged = False

    def export_structure(self) -> str:
        """
        Write the structure.txt file
        """

        if not self.subproblems:
            raise RuntimeError("Subproblem list must have at least one sub problem")

        # A mapping similar to the Xpansion mapping for keeping track of variable indexes
        # in Master and Sub-problem files
        problem_to_candidates: Dict[str, Dict[str, int]] = {}
        candidates = set()

        problem_to_candidates["master"] = {}
        for solver_var_info in self.master.context._solver_variables.values():
            problem_to_candidates["master"][
                solver_var_info.name
            ] = solver_var_info.column_id
            candidates.add(solver_var_info.name)

        for problem in self.subproblems:
            problem_to_candidates[problem.name] = {}

            for solver_var_info in problem.context._solver_variables.values():
                if solver_var_info.name in candidates:
                    # If candidate was identified in master
                    problem_to_candidates[problem.name][
                        solver_var_info.name
                    ] = solver_var_info.column_id

        structure_str = ""
        for problem_name, candidate_to_index in problem_to_candidates.items():
            for candidate, index in candidate_to_index.items():
                structure_str += f"{problem_name:>50}{candidate:>50}{index:>10}\n"

        return structure_str

    def export_options(
        self, *, solver_name: str = "XPRESS", log_level: int = 0
    ) -> Dict[str, Any]:
        # Default values
        options_values = {
            "MAX_ITERATIONS": -1,
            "ABSOLUTE_GAP": 1,
            "RELATIVE_GAP": 1e-6,
            "RELAXED_GAP": 1e-5,
            "AGGREGATION": False,
            "OUTPUTROOT": ".",
            "TRACE": True,
            "SLAVE_WEIGHT": "CONSTANT",
            "SLAVE_WEIGHT_VALUE": 1,
            "MASTER_NAME": f"{self.master.name}",
            "STRUCTURE_FILE": f"{self.structure_filename}",
            "INPUTROOT": ".",
            "CSV_NAME": "benders_output_trace",
            "BOUND_ALPHA": True,
            "SEPARATION_PARAM": 0.5,
            "BATCH_SIZE": 0,
            "JSON_FILE": f"{self.output_path}/out.json",
            "LAST_ITERATION_JSON_FILE": f"{self.output_path}/last_iteration.json",
            "MASTER_FORMULATION": "integer",
            "SOLVER_NAME": solver_name,
            "TIME_LIMIT": 1_000_000_000_000,
            "LOG_LEVEL": log_level,
            "LAST_MASTER_MPS": "master_last_iteration",
            "LAST_MASTER_BASIS": "master_last_basis.bss",
        }
        return options_values

    def initialise(
        self,
        *,
        solver_name: str = "XPRESS",
        log_level: int = 0,
        is_debug: bool = False,
    ) -> None:
        serialize(
            f"{self.master.name}.mps", self.master.export_as_mps(), self.emplacement
        )
        for subproblem in self.subproblems:
            serialize(
                f"{subproblem.name}.mps", subproblem.export_as_mps(), self.emplacement
            )
        serialize(
            f"{self.structure_filename}", self.export_structure(), self.emplacement
        )
        serialize_json(
            "options.json",
            self.export_options(solver_name=solver_name, log_level=log_level),
            self.emplacement,
        )

        if is_debug:
            serialize(
                f"{self.master.name}.lp", self.master.export_as_lp(), self.emplacement
            )
            for subproblem in self.subproblems:
                serialize(
                    f"{subproblem.name}.lp", subproblem.export_as_lp(), self.emplacement
                )

    def read_solution(self) -> None:
        try:
            data = read_json("out.json", self.emplacement / self.output_path)

        except FileNotFoundError:
            # TODO For now, it will return as if nothing is wrong
            # modify it with runner's run
            print("Return without reading it for now")
            return

        if self.is_merged:
            self.solution = BendersMergedSolution(data)
        else:
            self.solution = BendersDecomposedSolution(data)

    def run(
        self,
        *,
        solver_name: str = "XPRESS",
        log_level: int = 0,
        should_merge: bool = False,
        show_debug: bool = False,
    ) -> bool:
        self.initialise(
            solver_name=solver_name, log_level=log_level, is_debug=show_debug
        )

        if not should_merge:
            return_code = BendersRunner(self.emplacement).run()
        else:
            self.is_merged = True
            return_code = MergeMPSRunner(self.emplacement).run()

        if return_code == 0:
            self.read_solution()
            return True
        else:
            return False


def build_benders_decomposed_problem(
    decision_tree_root: DecisionTreeNode,
    database: DataBase,
    *,
    border_management: BlockBorderManagement = BlockBorderManagement.CYCLE,
    solver_id: str = "GLOP",
    struct_filename: str = "structure.txt",
) -> BendersDecomposedProblem:
    """
    Entry point to build the xpansion pathway problem.

    For each node of the tree, it builds a Master and a Subproblem.
    Then it defines a coupled problem that merges all masters along with
    its pathway constraints into one 'tree master' problem.

    Returns a Benders Decomposed problem
    """

    if not decision_tree_root.is_leaves_prob_sum_one():
        raise ValueError("Decision tree leaves' probability must sum one!")

    null_time_block = TimeBlock(
        0, [0]
    )  # Not necessary for master, but list must be non-empty
    null_scenario = 0  # Not necessary for master

    decision_tree_root._add_coupling_constraints()

    coupler = build_problem(
        decision_tree_root.coupling_network,
        database,
        null_time_block,
        null_scenario,
        problem_name="coupler",
        solver_id=solver_id,
        build_strategy=InvestmentProblemStrategy(),
        risk_strategy=ExpectedValue(0.0),
        use_full_var_name=False,
    )

    masters = []  # Benders Decomposed Master Problem
    subproblems = []  # Benders Decomposed Sub-problems

    for tree_node in decision_tree_root.traverse():
        suffix_tree = f"_{tree_node.id}" if decision_tree_root.size > 1 else ""

        masters.append(
            build_problem(
                tree_node.network,
                database,
                null_time_block,
                null_scenario,
                problem_name=f"master{suffix_tree}",
                solver_id=solver_id,
                build_strategy=InvestmentProblemStrategy(),
                decision_tree_node=tree_node.id,
                risk_strategy=ExpectedValue(tree_node.prob),
            )
        )

        for block in tree_node.config.blocks:
            suffix_block = f"_b{block.id}" if len(tree_node.config.blocks) > 1 else ""

            subproblems.append(
                build_problem(
                    tree_node.network,
                    database,
                    block,
                    tree_node.config.scenarios,
                    problem_name=f"subproblem{suffix_tree}{suffix_block}",
                    solver_id=solver_id,
                    build_strategy=OperationalProblemStrategy(),
                    decision_tree_node=tree_node.id,
                    risk_strategy=ExpectedValue(tree_node.prob),
                )
            )

    master = fusion_problems(masters, coupler)

    return BendersDecomposedProblem(
        master, subproblems, struct_filename=struct_filename
    )
