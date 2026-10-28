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

import gc
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
from gems_craft.study.parsing import OutputFormat
from gems_craft.study.study import Study
from gems_runner.simulation.heuristic_runner import (
    apply_thermal_heuristics,
    should_apply_heuristics,
)
from gems_runner.simulation.optimization import OptimizationProblem, build_problem
from gems_runner.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
    _concat_frames,
)
from gems_runner.simulation.table_folder import _TableWriter
from gems_runner.simulation.table_schema import _empty_frame
from gems_runner.simulation.time_block import TimeBlock

SIMULATION_TABLE_FOLDER = "simulation_table"


class SimulationSession:
    """Solves a study and returns its results as a ``SimulationTable``.

    Without *output_dir*, the results stay in memory. With it, the rows of each
    block are passed to ``output_dir / "simulation_table"`` as soon as it is
    solved, and ``run()`` returns the stored table (see
    docs/user-guide/outputs.md).
    """

    def __init__(
        self,
        study: Study,
        optim_config: OptimConfig,
        run_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
        output_format: OutputFormat = OutputFormat.PARQUET,
    ) -> None:
        self.study = study
        self.optim_config = optim_config
        self.run_id = run_id or str(uuid4())
        self.output_dir = output_dir
        self.output_format = OutputFormat(output_format)
        self._apply_heuristics = should_apply_heuristics(study)

    @property
    def scenario_ids(self) -> List[int]:
        return self.optim_config.scenario_scope.scenario_ids

    def run(self) -> SimulationTable:
        """Entry point. Validates the optim config, solves the study and returns
        its results: the stored table with an ``output_dir``, a table in memory
        otherwise, and an empty table in Benders mode."""
        validate_optim_config(
            self.optim_config, self.study.system, self.study.scenario_builder
        )
        mode = self.optim_config.resolution.mode
        if mode == ResolutionMode.BENDERS_DECOMPOSITION:
            self._run_benders()
            # Benders produces no simulation table: no table folder either.
            return SimulationTable(
                _empty_frame(), table_id=self.run_id, scenarios=self.scenario_ids
            )
        if mode not in (
            ResolutionMode.FRONTAL,
            ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            ResolutionMode.PARALLEL_SUBPROBLEMS,
        ):
            raise ValueError(f"Unknown resolution mode: {mode}")
        first = self.optim_config.time_scope.first_time_step
        last = self.optim_config.time_scope.last_time_step
        if mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS and first == last:
            # Checked before anything is solved or written: the sequential
            # loop solves no block on a single time step. Remove once the
            # block list is built upfront.
            raise ValueError(
                "Sequential mode solves no time block when the time scope has a "
                f"single time step ({first}..{last}): use frontal or "
                "parallel-subproblems mode."
            )

        if self.output_dir is None:
            # In memory: the whole table is returned, so it is built at once.
            # Frontal solves a single block holding every scenario: one call
            # gives the table, with the rows and order of build(). The
            # sub-problem modes concatenate the rows of their blocks.
            # On disk (below), rows are written as they are solved instead,
            # so the full table is never held.
            if mode == ResolutionMode.FRONTAL:
                _, rows = self._run_block(
                    self._frontal_block(), scenario_ids=self.scenario_ids
                )
            else:
                rows = _concat_frames(
                    r for _, blocks in self._iter_scenario_blocks(mode) for r in blocks
                )
            return SimulationTable(
                rows, table_id=self.run_id, scenarios=self.scenario_ids
            )

        # Resolved once, so that the table is opened where it was written even
        # if the working directory changes during the run.
        folder = Path(self.output_dir).resolve() / SIMULATION_TABLE_FOLDER
        writer = _TableWriter(folder, self.run_id, self.output_format)
        try:
            for scenario, blocks in self._iter_scenario_blocks(mode):
                if scenario is None:
                    writer.write_common(_concat_frames(blocks))
                    continue
                writer.start_scenario(scenario)
                # Each block is passed on as soon as it is solved.
                for block_dataframe in blocks:
                    writer.append(block_dataframe)
                    del block_dataframe  # released before the next block is solved
                writer.end_scenario()  # before the next scenario is solved
                if mode != ResolutionMode.FRONTAL:
                    # The solved problems hold reference cycles: free them now
                    # rather than when the garbage collector runs.
                    gc.collect()
        except BaseException:  # Ctrl-C too
            # The completed scenarios stay, without a manifest; the unfinished
            # one is removed, so that no file is left open or half written.
            writer.drop_open_scenario()
            raise
        writer.finalize()  # the manifest last: the folder is complete
        return SimulationTable.open(folder)

    # ------------------------------------------------------------------
    # Resolution strategies: the rows of the run, block by block
    # ------------------------------------------------------------------

    def _iter_scenario_blocks(
        self, mode: ResolutionMode
    ) -> Iterator[Tuple[Optional[int], Iterator[pd.DataFrame]]]:
        """Yield ``(scenario, blocks)`` for each scenario of the run, where
        *blocks* solves and yields the rows of the scenario's blocks, and
        ``(None, blocks)`` first for rows shared by all scenarios. *blocks* must
        be consumed before the next item is asked for. Private: results are
        returned by ``run()`` only.

        Sequential and parallel modes yield the rows of each block as soon as
        it is solved. Frontal mode solves once, then yields the shared rows and
        the rows of each scenario, built one scenario at a time."""
        if mode == ResolutionMode.FRONTAL:
            yield from self._iter_frontal()
        elif mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
            for scenario_id in self.scenario_ids:
                yield scenario_id, self._iter_sequential(scenario_id)
        else:
            for scenario_id in self.scenario_ids:
                yield scenario_id, self._iter_parallel(scenario_id)

    def _frontal_block(self) -> TimeBlock:
        return TimeBlock(
            0,
            list(
                range(
                    self.optim_config.time_scope.first_time_step,
                    self.optim_config.time_scope.last_time_step + 1,
                )
            ),
        )

    def _iter_frontal(self) -> Iterator[Tuple[Optional[int], Iterator[pd.DataFrame]]]:
        # The rows of each scenario are built from the solution arrays one at a
        # time: the full table is never built.
        problem = self._solve_block(self._frontal_block(), self.scenario_ids)
        for scenario, rows in SimulationTableBuilder()._iter_scenario_frames(
            problem, self.scenario_ids
        ):
            yield scenario, iter([rows])
            del rows

    def _iter_sequential(self, scenario_id: int) -> Iterator[pd.DataFrame]:
        """Solve the blocks of *scenario_id* one after the other, each starting
        from the end of the previous one, and yield the rows of each block."""
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]
        block_overlap: int = cfg.block_overlap
        carry_over_length: int = cfg.effective_carry_over_length

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
            problem, block_dataframe = self._run_block(
                block,
                scenario_ids=[scenario_id],
                initial_values=carry_over or None,
            )
            yield block_dataframe
            # Written by the caller: not held during the next solve.
            del block_dataframe
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

    def _iter_parallel(self, scenario_id: int) -> Iterator[pd.DataFrame]:
        """Solve the blocks of *scenario_id* independently of each other, and
        yield the rows of each block."""
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]
        last = self.optim_config.time_scope.last_time_step
        starts = range(
            self.optim_config.time_scope.first_time_step, last + 1, block_length
        )
        for i, t in enumerate(starts):
            block = TimeBlock(i, list(range(t, min(t + block_length, last + 1))))
            yield self._run_block(block, scenario_ids=[scenario_id])[1]

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
    # Solving one block
    # ------------------------------------------------------------------

    def _run_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> Tuple[OptimizationProblem, pd.DataFrame]:
        """Build and solve one block, then convert it to rows.

        Returns both the solved problem (for carry-over extraction or inspection)
        and its rows with correct absolute-time and scenario indices.
        scenario_ids_remap equals scenario_ids because the list of MC scenario IDs
        IS the mapping from internal 0-based position to actual MC identifier.
        """
        problem = self._solve_block(block, scenario_ids, initial_values)
        rows = SimulationTableBuilder()._build_frame(
            problem, scenario_ids_remap=scenario_ids
        )
        return problem, rows

    def _solve_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> OptimizationProblem:
        """Build and solve one block."""
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
