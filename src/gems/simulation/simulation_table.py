from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from attr import dataclass

# from gems.simulation.extra_output import ExtraOutput
# from optimization import OptimizationProblem  # Adjust import as needed
from gems.simulation.output_values import OutputValues


class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute-time-index"
    BLOCK_TIME_INDEX = "block-time-index"
    SCENARIO_INDEX = "scenario-index"
    VALUE = "value"
    BASIS_STATUS = "basis-status"


class SimulationTableBuilder:
    """Builds simulation tables from solver output values."""

    def __init__(self, simulation_id: Optional[str] = None) -> None:
        self.simulation_id: str = simulation_id or datetime.now().strftime(
            "%Y%m%d-%H%M"
        )

    def build(
        self, output_values: OutputValues, absolute_time_offset: Optional[int] = None
    ) -> pd.DataFrame:
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")

        context = output_values.problem.context
        block = context._block.id
        block_size = context.block_length()

        absolute_time_offset = absolute_time_offset or (block - 1) * block_size
        assert absolute_time_offset is not None

        rows: list[dict[str, Any]] = []
        rows += self._collect_solver_outputs(output_values, block, absolute_time_offset)
        rows += self._collect_extra_outputs(output_values, block, absolute_time_offset)
        rows.append(self._collect_objective_value(output_values, block))

        return pd.DataFrame(rows, columns=[c.value for c in SimulationColumns])

    # -------------------------------------------------------------------------
    # Solver outputs
    # -------------------------------------------------------------------------
    def _collect_solver_outputs(
        self, output_values: OutputValues, block: int, abs_offset: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for comp_id, comp in output_values._components.items():
            if hasattr(comp, "_variables"):
                for var in comp._variables.values():
                    if hasattr(var, "_value"):
                        for ts_index, val in var._value.items():
                            basis_status = None
                            if hasattr(var, "_basis_status"):
                                if isinstance(var._basis_status, str):
                                    basis_status = var._basis_status
                                elif isinstance(var._basis_status, dict):
                                    basis_status = var._basis_status.get(ts_index)

                            var_name = var._name if hasattr(var, "_name") else ""
                            rows.append(
                                {
                                    SimulationColumns.BLOCK.value: block,
                                    SimulationColumns.COMPONENT.value: comp_id,
                                    SimulationColumns.OUTPUT.value: var_name,
                                    SimulationColumns.ABSOLUTE_TIME_INDEX.value: abs_offset
                                    + ts_index.time,
                                    SimulationColumns.BLOCK_TIME_INDEX.value: ts_index.time,
                                    SimulationColumns.SCENARIO_INDEX.value: ts_index.scenario,
                                    SimulationColumns.VALUE.value: val,
                                    SimulationColumns.BASIS_STATUS.value: basis_status,
                                }
                            )
        return rows

    # -------------------------------------------------------------------------
    # Extra outputs
    # -------------------------------------------------------------------------
    def _collect_extra_outputs(
        self, output_values: OutputValues, block: int, abs_offset: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for comp_id, comp in output_values._components.items():
            if hasattr(comp, "_extra_outputs"):
                for name, extra_output in comp._extra_outputs.items():
                    for ts_index, val in extra_output._value.items():
                        # if hasattr(extra_output, "values"):
                        #     for ts_index, val in extra_output.values.items():
                        t: int = ts_index.time
                        s: int = ts_index.scenario
                        rows.append(
                            {
                                SimulationColumns.BLOCK.value: block,
                                SimulationColumns.COMPONENT.value: comp_id,
                                SimulationColumns.OUTPUT.value: name,
                                SimulationColumns.ABSOLUTE_TIME_INDEX.value: abs_offset
                                + t,
                                SimulationColumns.BLOCK_TIME_INDEX.value: t,
                                SimulationColumns.SCENARIO_INDEX.value: s,
                                SimulationColumns.VALUE.value: val,
                                SimulationColumns.BASIS_STATUS.value: None,
                            }
                        )
        return rows

    # -------------------------------------------------------------------------
    # Objective value
    # -------------------------------------------------------------------------
    def _collect_objective_value(
        self, output_values: OutputValues, block: int
    ) -> dict[str, Any]:
        assert output_values.problem is not None, "OutputValues problem is not set"
        assert (
            output_values.problem.solver is not None
        ), "OutputValues problem.solver is not set"
        return {
            SimulationColumns.BLOCK.value: block,
            SimulationColumns.COMPONENT.value: None,
            SimulationColumns.OUTPUT.value: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX.value: None,
            SimulationColumns.BLOCK_TIME_INDEX.value: None,
            SimulationColumns.SCENARIO_INDEX.value: None,
            SimulationColumns.VALUE.value: output_values.problem.solver.Objective().Value(),
            SimulationColumns.BASIS_STATUS.value: None,
        }


@dataclass
class SimulationTableWriter:
    """Handles writing simulation tables to CSV."""

    simulation_table: pd.DataFrame

    def write_csv(
        self,
        output_dir: Union[str, Path],
        simulation_id: str,
        optim_nb: int,
    ) -> Path:
        """Write the simulation table to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"simulation_table_{simulation_id}_{optim_nb}.csv"
        filepath = output_dir / filename
        self.simulation_table.to_csv(filepath, index=False)
        return filepath
