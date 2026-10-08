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

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import xarray as xr

from gems_craft.optim_config.parsing import (
    OptimConfig,
    ResolutionMode,
)
from gems_craft.optim_config.validation import validate_optim_config
from gems_craft.study.folder import load_study
from gems_craft.study.study import Study
from gems_runner.simulation.heuristic_runner import (
    apply_thermal_heuristics,
    should_apply_heuristics,
)
from gems_runner.simulation.optimization import OptimizationProblem, build_problem
from gems_runner.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
    merge_simulation_tables,
)
from gems_runner.simulation.time_block import TimeBlock


class SimulationSession:
    def __init__(
        self,
        study: Study,
        optim_config: OptimConfig,
        run_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.study = study
        self.optim_config = optim_config
        self.run_id = run_id or str(uuid4())
        self.output_dir = output_dir
        self._apply_heuristics = should_apply_heuristics(study)

    @property
    def scenario_ids(self) -> List[int]:
        return self.optim_config.scenario_scope.scenario_ids

    def run(self) -> SimulationTable:
        """Solve and return the results of all scenarios as one table (an
        empty table in Benders mode, which produces no simulation table).

        Holds all scenarios in memory; use ``iter_scenario_tables`` to handle
        the results one scenario at a time.
        """
        tables = list(self.iter_scenario_tables())
        if not tables:
            return SimulationTable(pd.DataFrame(), table_id=self.run_id)
        return merge_simulation_tables(tables, table_id=self.run_id)

    def iter_scenario_tables(self) -> Iterator[SimulationTable]:
        """Solve and yield the results one scenario at a time.

        The optim-config is validated when this method is called; solving
        starts when the iteration starts, and a scenario is only solved once
        the previous result has been consumed.

        - Sequential and parallel subproblem modes yield each scenario as soon
          as it is solved.
        - Frontal mode solves all scenarios at once, then yields the rows
          shared by all scenarios (only when there are several scenarios, with
          an empty ``scenario_index``), then each scenario, built on demand
          from the solution: no table holding all scenarios is built.
        - Benders mode runs the decomposition and yields nothing, as it
          produces no simulation table.

        A scenario without rows of its own is not yielded.
        """
        validate_optim_config(
            self.optim_config, self.study.system, self.study.scenario_builder
        )
        mode = self.optim_config.resolution.mode
        if mode == ResolutionMode.FRONTAL:
            return self._iter_frontal()
        elif mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
            return self._iter_sequential()
        elif mode == ResolutionMode.PARALLEL_SUBPROBLEMS:
            return self._iter_parallel()
        elif mode == ResolutionMode.BENDERS_DECOMPOSITION:
            return self._iter_benders()
        raise ValueError(f"Unknown resolution mode: {mode}")

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _iter_frontal(self) -> Iterator[SimulationTable]:
        block = TimeBlock(
            0,
            list(
                range(
                    self.optim_config.time_scope.first_time_step,
                    self.optim_config.time_scope.last_time_step + 1,
                )
            ),
        )
        problem = self._solve_block(block, scenario_ids=self.scenario_ids)
        yield from SimulationTableBuilder().iter_scenario_tables(
            problem, scenario_ids_remap=self.scenario_ids, table_id=self.run_id
        )

    def _iter_sequential(self) -> Iterator[SimulationTable]:
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]
        block_overlap: int = cfg.block_overlap
        carry_over_length: int = cfg.effective_carry_over_length

        for scenario_id in self.scenario_ids:
            scenario_tables: List[SimulationTable] = []
            t_start = self.optim_config.time_scope.first_time_step
            block_id = 0
            carry_over: Dict[Tuple[str, str], xr.DataArray] = {}

            while t_start < self.optim_config.time_scope.last_time_step:
                end = min(
                    t_start + block_length,
                    self.optim_config.time_scope.last_time_step + 1,
                )
                timesteps = list(range(t_start, end))
                block = TimeBlock(block_id, timesteps)
                problem, table = self._run_block(
                    block,
                    scenario_ids=[scenario_id],
                    initial_values=carry_over or None,
                )
                scenario_tables.append(table)
                # Block N and block N+1 share `block_overlap` absolute
                # timesteps: block N's local indices `block_length - overlap
                # ...` are block N+1's local indices `0 ...`.
                delta = block_length - block_overlap
                t_start += delta
                carry_over = self._extract_carry_over(
                    problem,
                    local_start=delta,
                    length=carry_over_length,
                )
                block_id += 1
            yield from self._merge_blocks(scenario_tables)

    def _iter_parallel(self) -> Iterator[SimulationTable]:
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]

        for scenario_id in self.scenario_ids:
            scenario_tables: List[SimulationTable] = []
            starts = range(
                self.optim_config.time_scope.first_time_step,
                self.optim_config.time_scope.last_time_step + 1,
                block_length,
            )
            blocks = [
                TimeBlock(
                    i,
                    list(
                        range(
                            t,
                            min(
                                t + block_length,
                                self.optim_config.time_scope.last_time_step + 1,
                            ),
                        )
                    ),
                )
                for i, t in enumerate(starts)
            ]
            for block in blocks:
                _, table = self._run_block(block, scenario_ids=[scenario_id])
                scenario_tables.append(table)
            yield from self._merge_blocks(scenario_tables)

    def _iter_benders(self) -> Iterator[SimulationTable]:
        """Run Benders decomposition; yields nothing, as Benders writes its
        results itself and produces no simulation table."""
        self._run_benders()
        yield from ()

    def _run_benders(self) -> None:
        from gems_runner.simulation import (
            BendersRunner,
            build_couplings,
            build_decomposed_problems,
            dump_couplings,
        )

        block = TimeBlock(
            1,
            list(
                range(
                    self.optim_config.time_scope.first_time_step,
                    self.optim_config.time_scope.last_time_step + 1,
                )
            ),
        )
        decomposed = build_decomposed_problems(
            self.study, block, self.scenario_ids, self.optim_config
        )

        if decomposed.master is not None and self.output_dir is not None:
            dump_couplings(
                build_couplings(decomposed, self.optim_config), self.output_dir
            )
            BendersRunner(emplacement=self.output_dir).run()
        else:
            raise RuntimeError(
                "Benders decomposition requires a master problem and an output directory for coupling files."
            )

    # ------------------------------------------------------------------
    # Map / reduce helpers
    # ------------------------------------------------------------------

    def _run_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> Tuple[OptimizationProblem, SimulationTable]:
        """MAP: build and solve one block, then convert to a SimulationTable.

        Returns both the solved problem (for carry-over extraction or inspection)
        and the SimulationTable with correct absolute-time and scenario indices.
        scenario_ids_remap equals scenario_ids because the list of MC scenario IDs
        IS the mapping from internal 0-based position to actual MC identifier.
        """
        try:
            problem = self._solve_block(block, scenario_ids, initial_values)
        except Exception as error:
            # The errors raised while building or solving name neither the
            # scenario nor the block.
            error.add_note(
                f"While solving scenario {', '.join(map(str, scenario_ids))}, "
                f"block {block.id} "
                f"(time steps {block.timesteps[0]}-{block.timesteps[-1]})."
            )
            raise
        table = SimulationTableBuilder().build(
            problem, scenario_ids_remap=scenario_ids, table_id=self.run_id
        )
        return problem, table

    def _solve_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> OptimizationProblem:
        """Build and solve one block (plus the heuristic re-solve, if any)."""
        problem = build_problem(
            self.study,
            block,
            scenario_ids,
            optim_config=self.optim_config,
            initial_values=initial_values,
        )
        solver_name = self.optim_config.solver_options.name
        # Explicit solver parameters take precedence over the logs switch.
        solver_kwargs = {
            **OptimizationProblem.log_output_options(
                solver_name, self.optim_config.solver_options.logs
            ),
            **self.optim_config.solver_options.parsed_parameters(),
        }
        problem.solve(solver_name=solver_name, **solver_kwargs)
        self._check_solved(problem)
        if self._apply_heuristics:
            apply_thermal_heuristics(problem, self.optim_config, scenario_ids)
            problem.solve(solver_name=solver_name, **solver_kwargs)
            self._check_solved(problem)
        return problem

    @staticmethod
    def _check_solved(problem: OptimizationProblem) -> None:
        if problem.status != "ok":
            raise RuntimeError(
                f"Problem {problem.name!r} was not solved to optimality "
                f"(termination_condition={problem.termination_condition!r})."
            )

    def _merge_blocks(
        self, block_tables: List[SimulationTable]
    ) -> Iterator[SimulationTable]:
        """REDUCE: merge one scenario's block tables into its table; yields
        nothing when no block was solved."""
        if block_tables:
            yield merge_simulation_tables(block_tables, table_id=self.run_id)

    @staticmethod
    def _extract_carry_over(
        problem: OptimizationProblem,
        local_start: int,
        length: int,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Extract variable values over *length* timesteps starting at *local_start*.

        The returned arrays keep a ``time`` dimension re-indexed to
        ``0 .. length-1`` so they align with the leading timesteps of the next
        block's variables.  The window is clamped to the solved block's actual
        horizon (a truncated final block can be shorter than ``block_length``),
        so fewer than *length* values may be carried over.

        Only variables carrying a ``time`` dimension are extracted.
        Time-independent variables (e.g. an investment capacity) are
        deliberately left free in every block, so each block re-optimizes them
        independently.
        """
        carry_over: Dict[Tuple[str, str], xr.DataArray] = {}
        if length <= 0 or problem.linopy_model.solution is None:
            return carry_over
        for (model, var_name), linopy_var in problem._linopy_vars.items():
            if "time" in linopy_var.dims:
                sol_da = problem.get_variable_solution(model, var_name)
                if sol_da is not None:
                    window = sol_da.isel(time=slice(local_start, local_start + length))
                    if window.sizes["time"] == 0:
                        continue
                    carry_over[(model, var_name)] = window.assign_coords(
                        time=list(range(window.sizes["time"]))
                    )
        return carry_over
