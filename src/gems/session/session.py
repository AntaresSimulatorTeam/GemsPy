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

from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import xarray as xr

from gems.optim_config.parsing import OptimConfig, ResolutionMode, load_optim_config
from gems.simulation.optimization import OptimizationProblem, build_problem
from gems.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
    merge_simulation_tables,
)
from gems.simulation.time_block import TimeBlock
from gems.study.folder import load_study
from gems.study.study import Study


@dataclass
class SimulationSession:
    study: Study
    optim_config: OptimConfig
    executor: Optional[Executor] = None
    materialize_per_worker: bool = False
    run_id: str = field(default_factory=lambda: str(uuid4()))
    output_dir: Optional[Path] = None

    @property
    def scenario_ids(self) -> List[int]:
        return list(range(self.optim_config.scenario_scope.nb_scenarios))

    def run(self) -> SimulationTable:
        """Entry point. Dispatches to the appropriate resolution strategy."""
        mode = self.optim_config.resolution.mode
        if mode == ResolutionMode.FRONTAL:
            return self._run_frontal()
        elif mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
            return self._run_sequential()
        elif mode == ResolutionMode.PARALLEL_SUBPROBLEMS:
            return self._run_parallel()
        elif mode == ResolutionMode.BENDERS_DECOMPOSITION:
            return self._run_benders()
        raise ValueError(f"Unknown resolution mode: {mode}")

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _run_frontal(self) -> SimulationTable:
        block = TimeBlock(
            0,
            list(
                range(
                    self.optim_config.time_scope.first_time_step,
                    self.optim_config.time_scope.last_time_step + 1,
                )
            ),
        )
        _, table = self._run_block(block, scenario_ids=self.scenario_ids)
        return table

    def _run_sequential(self) -> SimulationTable:
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]
        block_overlap: int = cfg.block_overlap

        def _run_one_scenario_sequential(scenario_id: int) -> SimulationTable:
            effective_study = self.study
            if self.materialize_per_worker:
                mat_db = self.study.database.materialize([scenario_id])
                effective_study = replace(self.study, database=mat_db)
            t_start = self.optim_config.time_scope.first_time_step
            block_id = 0
            carry_over: Dict[Tuple[str, str], xr.DataArray] = {}
            block_tables: List[SimulationTable] = []
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
                    study=effective_study,
                )
                block_tables.append(table)
                carry_over = self._extract_carry_over(problem, local_index=len(timesteps) - 1)
                t_start += block_length - block_overlap
                block_id += 1
            return self._reduce(block_tables)

        if self.executor is not None:
            futures = [
                self.executor.submit(_run_one_scenario_sequential, sid) for sid in self.scenario_ids
            ]
            scenario_tables = [f.result() for f in futures]
        else:
            scenario_tables = [_run_one_scenario_sequential(sid) for sid in self.scenario_ids]

        return self._reduce(scenario_tables)

    def _run_parallel(self, blocks_per_batch: int = 1) -> SimulationTable:
        cfg = self.optim_config.resolution
        block_length: int = cfg.block_length  # type: ignore[assignment]

        t_end = self.optim_config.time_scope.last_time_step + 1
        all_block_starts = list(
            range(self.optim_config.time_scope.first_time_step, t_end, block_length)
        )

        # Build batches: each batch is a list of independent (TimeBlock, scenario_ids) pairs
        # that will be executed on the same worker.
        batches: List[List[Tuple[TimeBlock, List[int]]]] = []
        for scenario_id in self.scenario_ids:
            for i in range(0, len(all_block_starts), blocks_per_batch):
                batch = []
                for block_idx, bs in enumerate(all_block_starts[i : i + blocks_per_batch], start=i):
                    timesteps = list(range(bs, min(bs + block_length, t_end)))
                    batch.append((TimeBlock(block_idx, timesteps), [scenario_id]))
                batches.append(batch)

        if self.executor is not None:
            futures = [self.executor.submit(self._run_batch, batch) for batch in batches]
            tables = [t for f in futures for t in f.result()]
        else:
            tables = [t for batch in batches for t in self._run_batch(batch)]

        return self._reduce(tables)

    def _run_benders(self) -> SimulationTable:
        import pandas as pd

        from gems.simulation import (
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
            dump_couplings(build_couplings(decomposed, self.optim_config), self.output_dir)
            BendersRunner(emplacement=self.output_dir).run()
        else:
            raise RuntimeError(
                "Benders decomposition requires a master problem and an output directory for coupling files."
            )
        return SimulationTable(pd.DataFrame())

    # ------------------------------------------------------------------
    # Map / reduce helpers
    # ------------------------------------------------------------------

    def _run_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
        study: Optional[Study] = None,
    ) -> Tuple[OptimizationProblem, SimulationTable]:
        """MAP: build and solve one block, then convert to a SimulationTable.

        Returns both the solved problem (for carry-over extraction or inspection)
        and the SimulationTable with correct absolute-time and scenario indices.
        scenario_ids_remap equals scenario_ids because the list of MC scenario IDs
        IS the mapping from internal 0-based position to actual MC identifier.
        """
        effective_study = study if study is not None else self.study
        problem = build_problem(
            effective_study,
            block,
            scenario_ids,
            optim_config=self.optim_config,
            initial_values=initial_values,
        )
        problem.solve(
            solver_name=self.optim_config.solver_options.name,
            solver_logs=self.optim_config.solver_options.logs,
            **self.optim_config.solver_options.parsed_parameters(),
        )
        table = SimulationTableBuilder().build(
            problem, scenario_ids_remap=scenario_ids, table_id=self.run_id
        )
        return problem, table

    def _run_batch(self, batch: List[Tuple[TimeBlock, List[int]]]) -> List[SimulationTable]:
        """Run a list of independent (block, scenario_ids) pairs on one worker."""
        study = self.study
        if self.materialize_per_worker:
            scenario_id = batch[0][1][0]
            mat_db = study.database.materialize([scenario_id])
            study = replace(study, database=mat_db)
        return [self._run_block(block, sids, study=study)[1] for block, sids in batch]

    def _reduce(self, tables: List[SimulationTable]) -> SimulationTable:
        """REDUCE: merge SimulationTables from one scenario's blocks into one."""
        return merge_simulation_tables(tables, table_id=self.run_id)

    @staticmethod
    def _extract_carry_over(
        problem: OptimizationProblem,
        local_index: int,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Extract variable values at *local_index* for use as initial values in the next block."""
        carry_over: Dict[Tuple[str, str], xr.DataArray] = {}
        solution = problem.linopy_model.solution
        if solution is None:
            return carry_over
        for (model, var_name), linopy_var in problem._linopy_vars.items():
            if "time" in linopy_var.dims and linopy_var.name in solution:
                sol_da: xr.DataArray = solution[linopy_var.name]
                carry_over[(model, var_name)] = sol_da.isel(time=local_index, drop=True)
        return carry_over
