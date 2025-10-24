# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

import pandas as pd

from gems.simulation.output_values import OutputValues
from gems.simulation.simulation_table import (
    SimulationColumns,
    SimulationTableBuilder,
    SimulationTableWriter,
)


@dataclass(frozen=True)
class FakeTimeIndex:
    """Represents a (time, scenario) pair for testing."""

    time: int
    scenario: int


@dataclass(frozen=True)
class FakeVariable:
    """Mimics a solver variable with values and basis statuses."""

    _name: str
    _value: dict[FakeTimeIndex, float]
    _basis_status: dict[FakeTimeIndex, str]


@dataclass(frozen=True)
class FakeComponent:
    """Container for fake variables."""

    _variables: dict[str, FakeVariable]


@dataclass(frozen=True)
class FakeSolver:
    """Fake solver providing a fixed objective value."""

    @dataclass(frozen=True)
    class Obj:
        def Value(self) -> float:
            return 42.0

    def Objective(self):
        return self.Obj()


@dataclass(frozen=True)
class FakeContext:
    """Fake optimization context with a single block and block length."""

    @dataclass(frozen=True)
    class Block:
        id: int = 1

    _block: Block = Block()

    def block_length(self) -> int:
        return 3


@dataclass(frozen=True)
class FakeProblem:
    """Fake problem that binds context and solver."""

    context: FakeContext = FakeContext()
    solver: FakeSolver = FakeSolver()


class FakeOutputValues(OutputValues):
    """Simplified OutputValues holding fake components and optional extras."""

    def __init__(self, problem: FakeProblem, components: dict, extra_outputs=None):
        self.problem = problem  # type: ignore
        self._components = components
        self._extra_outputs = extra_outputs or {}


def test_simulation_table_builder_manual(tmp_path):
    """Test SimulationTableBuilder and SimulationTableWriter with fake data."""
    problem = FakeProblem()

    ts0 = FakeTimeIndex(time=0, scenario=0)
    ts1 = FakeTimeIndex(time=1, scenario=0)

    var = FakeVariable(
        _name="p",
        _value={ts0: 10.0, ts1: 20.0},
        _basis_status={ts0: "BASIC", ts1: "NONBASIC"},
    )

    component = FakeComponent(_variables={"var1": var})
    output_values = FakeOutputValues(problem=problem, components={"compA": component})

    builder = SimulationTableBuilder(simulation_id="test")
    df = builder.build(output_values)  # type: ignore

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
        df.reset_index(drop=True).fillna(pd.NA),
        expected_df,
        check_dtype=False,
    )

    writer = SimulationTableWriter(df)
    csv_path = writer.write_csv(tmp_path, simulation_id="test", optim_nb=1)

    assert csv_path.exists(), "CSV file was not created"

    with csv_path.open("r") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(col.value for col in SimulationColumns)
    assert first_line == expected_header, "CSV header does not match expected columns"

    csv_path.unlink()
