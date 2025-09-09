# Standard library imports
from pathlib import Path

# Third-party imports
import pandas as pd
import pytest

# Local application/library imports
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.simulation import OutputValues, TimeBlock, build_problem
from gems.simulation.simulation_table import (
    SimulationColumns,
    SimulationTableBuilder,
    SimulationTableWriter,
)
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import build_data_base, build_network, resolve_system


class FakeTimeIndex:
    def __init__(self, time: int, scenario: int):
        self.time = time
        self.scenario = scenario


class FakeSolver:
    def Objective(self):
        class Obj:
            def Value(self) -> float:
                return 42.0

        return Obj()


class FakeContext:
    def __init__(self):
        class Block:
            id = 1

        self._block = Block()

    def block_length(self):
        return 3


class FakeProblem:
    def __init__(self):
        self.context = FakeContext()
        self.solver = FakeSolver()


class FakeVariable:
    def __init__(self, name, values, basis_status):
        self._name = name
        self._value = values
        self._basis_status = basis_status


class FakeComponent:
    def __init__(self, variables):
        self._variables = variables


class FakeOutputValues(OutputValues):
    def __init__(self, problem, components):
        self.problem = problem
        self._components = components


def test_simulation_table_builder_manual(tmp_path):
    problem = FakeProblem()

    ts0 = FakeTimeIndex(time=0, scenario=0)
    ts1 = FakeTimeIndex(time=1, scenario=0)

    var = FakeVariable(
        name="p",
        values={ts0: 10.0, ts1: 20.0},
        basis_status={ts0: "BASIC", ts1: "NONBASIC"},
    )

    component = FakeComponent({"var1": var})
    output_values = FakeOutputValues(problem, {"compA": component})

    # --- Build table ---
    builder = SimulationTableBuilder(simulation_id="test")
    df = builder.build(output_values)

    expected_rows = [
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 0,
            SimulationColumns.BLOCK_TIME_INDEX: 0,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 10.0,
            SimulationColumns.BASIS_STATUS: "BASIC",
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 1,
            SimulationColumns.BLOCK_TIME_INDEX: 1,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 20.0,
            SimulationColumns.BASIS_STATUS: "NONBASIC",
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: pd.NA,
            SimulationColumns.OUTPUT: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX: pd.NA,
            SimulationColumns.BLOCK_TIME_INDEX: pd.NA,
            SimulationColumns.SCENARIO_INDEX: pd.NA,
            SimulationColumns.VALUE: 42.0,
            SimulationColumns.BASIS_STATUS: pd.NA,
        },
    ]
    expected_df = pd.DataFrame(expected_rows).fillna(pd.NA)

    pd.testing.assert_frame_equal(
        df.reset_index(drop=True).fillna(pd.NA), expected_df, check_dtype=False
    )

    writer = SimulationTableWriter(df)
    csv_path = writer.write_csv(tmp_path, simulation_id="test", optim_nb=1)

    assert csv_path.exists(), "CSV file was not created"

    with csv_path.open("r") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(col.value for col in SimulationColumns)
    assert first_line == expected_header, "CSV header does not match expected columns"
    csv_path.unlink()
