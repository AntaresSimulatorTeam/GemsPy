from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd

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
        self.simulation_id = simulation_id or datetime.now().strftime("%Y%m%d-%H%M")

    def build(self, output_values: OutputValues) -> pd.DataFrame:
        """Populate a DataFrame from OutputValues."""
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")

        context = output_values.problem.context
        block = context._block.id
        block_size = context.block_length()
        absolute_time_offset = (block - 1) * block_size

        rows = []

        for component_id, output_component in output_values._components.items():
            for _, var in output_component._variables.items():
                for ts_index, value in var._value.items():
                    basis_status = (
                        var._basis_status
                        if isinstance(var._basis_status, str)
                        else var._basis_status.get(ts_index)
                    )
                    row = {
                        SimulationColumns.BLOCK.value: block,
                        SimulationColumns.COMPONENT.value: component_id,
                        SimulationColumns.OUTPUT.value: var._name,
                        SimulationColumns.ABSOLUTE_TIME_INDEX.value: absolute_time_offset
                        + ts_index.time,
                        SimulationColumns.BLOCK_TIME_INDEX.value: ts_index.time,
                        SimulationColumns.SCENARIO_INDEX.value: ts_index.scenario,
                        SimulationColumns.VALUE.value: value,
                        SimulationColumns.BASIS_STATUS.value: basis_status,
                    }
                    rows.append(row)

        df = pd.DataFrame(rows, columns=[col.value for col in SimulationColumns])

        # Append objective value
        objective_value = output_values.problem.solver.Objective().Value()
        obj_row = {
            SimulationColumns.BLOCK.value: block,
            SimulationColumns.COMPONENT.value: None,
            SimulationColumns.OUTPUT.value: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX.value: None,
            SimulationColumns.BLOCK_TIME_INDEX.value: None,
            SimulationColumns.SCENARIO_INDEX.value: None,
            SimulationColumns.VALUE.value: objective_value,
            SimulationColumns.BASIS_STATUS.value: None,
        }
        df.loc[len(df)] = [obj_row.get(col.value, None) for col in SimulationColumns]

        return df

    def extra_output_eval(self) -> None:
        raise NotImplementedError("extra_output_eval() is not yet implemented.")

    def add_extra_output(self) -> None:
        raise NotImplementedError("add_extra_output() is not yet implemented.")


class SimulationTableWriter:
    """Handles writing simulation tables to CSV."""

    def __init__(self, simulation_table: pd.DataFrame) -> None:
        self.simulation_table = simulation_table

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
