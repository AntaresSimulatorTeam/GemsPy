# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Shared fakes/stubs for SimulationTableBuilder tests.

Used by test_simulation_table_mock.py, test_simulation_table_accessor.py and
test_simulation_table_export.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class FakeBlock:
    """Fake time block with an id and absolute timestep list."""

    id: int = 1
    timesteps: tuple = (0, 1)


@dataclass
class FakeLinopyVar:
    """Minimal linopy variable stub exposing name and component coords."""

    name: str
    coords: dict  # {"component": xr.DataArray}


@dataclass
class FakeModel:
    """Fake model with no extra outputs."""

    extra_outputs: dict = field(default_factory=dict)


@dataclass
class FakeStudy:
    model_components: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)


@dataclass
class FakeLinopyModel:
    """Fake linopy model exposing a solution dataset."""

    solution: dict  # lv.name -> xr.DataArray

    @property
    def dual(self) -> xr.Dataset:
        return xr.Dataset()

    solver_model = None


@dataclass
class FakeProblem:
    """Fake OptimizationProblem with the attributes used by SimulationTableBuilder."""

    block: FakeBlock = field(default_factory=FakeBlock)
    block_length: int = 2
    objective_value: float = 0.0
    linopy_model: Optional[FakeLinopyModel] = None
    _linopy_vars: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    model_components: dict = field(default_factory=dict)
    study: FakeStudy = field(default_factory=FakeStudy)
    scenarios: int = 1

    def get_variable_solution(
        self, model_id: object, var_name: str
    ) -> Optional[xr.DataArray]:
        lv = self._linopy_vars.get((model_id, var_name))
        if lv is None or self.linopy_model is None:
            return None
        return self.linopy_model.solution.get(lv.name)

    def get_variable_lower_bound(
        self, model_id: object, var_name: str
    ) -> Optional[xr.DataArray]:
        return None

    def get_variable_upper_bound(
        self, model_id: object, var_name: str
    ) -> Optional[xr.DataArray]:
        return None


def to_object_dtype(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast every column to numpy object dtype, normalising all nulls to None."""
    return pd.DataFrame(
        {col: frame[col].to_numpy(dtype=object, na_value=None) for col in frame.columns}
    )
