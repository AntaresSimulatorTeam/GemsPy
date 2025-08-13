from pathlib import Path
from typing import Optional, Union

import pandas as pd
from gems.simulation.output_values import OutputValues
from gems.study.data import TimeScenarioIndex
from datetime import datetime


class SimulationTable:
    simulation_id: str
    df: pd.DataFrame

    def __init__(self, simulation_id: Optional[str] = None, mode: str = "eco") -> None:
        # Unique simulation ID, from Antares logic or fallback to UUID
        if simulation_id:
            self.simulation_id = simulation_id
        else:
            now = datetime.now()
            self.simulation_id = now.strftime("%Y%m%d-%H%M") + mode

        self.df = pd.DataFrame(
            columns=[
                "simulation_id",
                "block",
                "component",
                "output",
                "absolute_time_index",
                "block_time_index",
                "scenario_index",
                "value",
                "basis_status",
            ]
        )

    def get_simulation_table(self) -> pd.DataFrame:
        return self.df

    def fill_from_output_values(
        self,
        output_values: OutputValues,
        block: int,
        absolute_time_offset: int,
        block_size: int,
        scenario_count: int,
    ) -> None:
        """
        Populate the SimulationTable from solver output values and append the objective value.
        """
        rows = []

        # Loop over components in OutputValues
        for comp_id, comp_obj in output_values._components.items():
            # Loop over variables for the component
            for var_name, var in comp_obj._variables.items():
                size_s, size_t = var._size  # (scenario_count, time_count)

                # Fallback if sizes don't match provided scenario_count/block_size
                if size_s == 0:
                    size_s = scenario_count
                if size_t == 0:
                    size_t = block_size

                for s in range(size_s):
                    for t in range(size_t):
                        value = var._value.get(TimeScenarioIndex(t, s), None)
                        basis_status = var._basis_status

                        rows.append(
                            {
                                "simulation_id": self.simulation_id,
                                "block": block,
                                "component": comp_id,
                                "output": var._name,
                                "absolute_time_index": absolute_time_offset + t,
                                "block_time_index": t + 1,
                                "scenario_index": s,
                                "value": value,
                                "basis_status": basis_status,
                            }
                        )

        # Store results as DataFrame
        self.df = pd.DataFrame(rows)

        # Ensure problem is set before accessing solver
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")
        objective_value = output_values.problem.solver.Objective().Value()

        # Append the objective value as the last row
        obj_row = {
            "simulation_id": self.simulation_id,
            "block": block,
            "component": None,
            "output": "OBJECTIVE_VALUE",
            "absolute_time_index": None,
            "block_time_index": None,
            "scenario_index": None,
            "value": objective_value,
            "basis_status": None,
        }
        self.df.loc[len(self.df)] = [obj_row.get(col, None) for col in self.df.columns]

    def extra_output_eval(self) -> None:
        raise NotImplementedError("extra_output_eval() is not yet implemented.")

    def add_extra_output(self) -> None:
        raise NotImplementedError("add_extra_output() is not yet implemented.")

    def write_csv(self, output_dir: Union[str, Path], optim_nb: int) -> Path:
        """Write the simulation table to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"simulation_table_{self.simulation_id}_{optim_nb}.csv"
        filepath = output_dir / filename
        self.df.to_csv(filepath, index=False)
        return filepath
