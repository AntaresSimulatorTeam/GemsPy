from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any, Dict

import pandas as pd
from attr import dataclass

from gems.expression.evaluate import EvaluationError, evaluate_expression
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

    def build(
        self,
        output_values: OutputValues,
        model_registry: Optional[Dict[str, Any]] = None,
        absolute_time_offset: Optional[int] = None,
    ) -> pd.DataFrame:
        """Populate a DataFrame from OutputValues, optionally adding extra outputs."""
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")

        context = output_values.problem.context
        block = context._block.id
        block_size = context.block_length()
        absolute_time_offset = absolute_time_offset or (block - 1) * block_size

        rows = []

        # === 1️⃣ Normal outputs ===
        for component_id, output_component in output_values._components.items():
            for _, var in output_component._variables.items():
                for ts_index, value in var._value.items():
                    basis_status = (
                        var._basis_status
                        if isinstance(var._basis_status, str)
                        else var._basis_status.get(ts_index)
                    )
                    row = {
                        SimulationColumns.BLOCK: block,
                        SimulationColumns.COMPONENT: component_id,
                        SimulationColumns.OUTPUT: var._name,
                        SimulationColumns.ABSOLUTE_TIME_INDEX: absolute_time_offset
                        + ts_index.time,
                        SimulationColumns.BLOCK_TIME_INDEX: ts_index.time,
                        SimulationColumns.SCENARIO_INDEX: ts_index.scenario,
                        SimulationColumns.VALUE: value,
                        SimulationColumns.BASIS_STATUS: basis_status,
                    }
                    rows.append(row)

        # === 2️⃣ Extra outputs ===
        if model_registry:
            for comp_id, comp in output_values._components.items():
                # find model for this component
                model_def = (
                    model_registry.get(comp_id)
                    or model_registry.get(comp_id.split(".")[-1])
                )
                if not model_def or not getattr(model_def, "extra_outputs", None):
                    continue

                # build variable context
                for ts_index in next(iter(comp._variables.values()))._value.keys():
                    context_vars = {
                        vname: float(vobj._value[ts_index])
                        for vname, vobj in comp._variables.items()
                        if ts_index in vobj._value and vobj._value[ts_index] is not None
                    }

                    for extra in model_def.extra_outputs:
                        # support both dicts and objects
                        if isinstance(extra, dict):
                            out_id = extra.get("id")
                            expr = extra.get("expression")
                        else:
                            out_id = getattr(extra, "id", None)
                            expr = getattr(extra, "expression", None)

                        if not out_id or expr is None:
                            continue

                        try:
                            val = evaluate_expression(expr, context_vars)
                        except EvaluationError:
                            val = float("nan")

                        row = {
                            SimulationColumns.BLOCK: block,
                            SimulationColumns.COMPONENT: comp_id,
                            SimulationColumns.OUTPUT: out_id,
                            SimulationColumns.ABSOLUTE_TIME_INDEX: absolute_time_offset + ts_index.time,
                            SimulationColumns.BLOCK_TIME_INDEX: ts_index.time,
                            SimulationColumns.SCENARIO_INDEX: ts_index.scenario,
                            SimulationColumns.VALUE: float(val),
                            SimulationColumns.BASIS_STATUS: None,
                        }
                        rows.append(row)


        # === 3️⃣ Objective value ===
        objective_value = output_values.problem.solver.Objective().Value()
        obj_row = {
            SimulationColumns.BLOCK: block,
            SimulationColumns.COMPONENT: None,
            SimulationColumns.OUTPUT: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX: None,
            SimulationColumns.BLOCK_TIME_INDEX: None,
            SimulationColumns.SCENARIO_INDEX: None,
            SimulationColumns.VALUE: objective_value,
            SimulationColumns.BASIS_STATUS: None,
        }
        rows.append(obj_row)

        df = pd.DataFrame(rows, columns=list(SimulationColumns))
        return df
    
    def build1(
        self, output_values: OutputValues, absolute_time_offset: Optional[int] = None
    ) -> pd.DataFrame:
        """Populate a DataFrame from OutputValues."""
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")

        context = output_values.problem.context
        block = context._block.id
        block_size = context.block_length()
        absolute_time_offset = absolute_time_offset or (block - 1) * block_size

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
                        SimulationColumns.BLOCK: block,
                        SimulationColumns.COMPONENT: component_id,
                        SimulationColumns.OUTPUT: var._name,
                        SimulationColumns.ABSOLUTE_TIME_INDEX: absolute_time_offset
                        + ts_index.time,
                        SimulationColumns.BLOCK_TIME_INDEX: ts_index.time,
                        SimulationColumns.SCENARIO_INDEX: ts_index.scenario,
                        SimulationColumns.VALUE: value,
                        SimulationColumns.BASIS_STATUS: basis_status,
                    }
                    rows.append(row)

        df = pd.DataFrame(rows, columns=list(SimulationColumns))

        # Append objective value
        objective_value = output_values.problem.solver.Objective().Value()
        obj_row = {
            SimulationColumns.BLOCK: block,
            SimulationColumns.COMPONENT: None,
            SimulationColumns.OUTPUT: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX: None,
            SimulationColumns.BLOCK_TIME_INDEX: None,
            SimulationColumns.SCENARIO_INDEX: None,
            SimulationColumns.VALUE: objective_value,
            SimulationColumns.BASIS_STATUS: None,
        }
        df.loc[len(df)] = [obj_row.get(col, None) for col in SimulationColumns]

        return df

    def extra_output_eval(self) -> None:
        raise NotImplementedError("extra_output_eval() is not yet implemented.")

    def add_extra_output(self) -> None:
        raise NotImplementedError("add_extra_output() is not yet implemented.")


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
