from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class Operation:
    type: Optional[str] = None
    multiply_by: Optional[Union[str, float]] = None
    divide_by: Optional[Union[str, float]] = None

    def execute(
        self,
        initial_value: Union[pd.Series, float],
        preprocessed_values: Optional[Union[dict[str, float], float]] = None,
    ) -> Union[float, pd.Series]:
        def resolve(value: Union[str, float]) -> Union[float, pd.Series]:
            if isinstance(value, str):
                if (
                    not isinstance(preprocessed_values, dict)
                    or value not in preprocessed_values
                ):
                    raise ValueError(
                        f"Missing value for key '{value}' in preprocessed_values"
                    )
                return preprocessed_values[value]
            return value

        if self.type == "max":
            return float(max(initial_value))  # type: ignore

        if self.multiply_by is not None:
            return initial_value * resolve(self.multiply_by)

        if self.divide_by is not None:
            return initial_value / resolve(self.divide_by)

        raise ValueError(
            "Operation must have at least one of 'multiply_by', 'divide_by', or 'type'"
        )


@dataclass(frozen=True)
class TimeseriesData:
    path: Path
    column: int
    operation: Optional[Operation] = None


@dataclass(frozen=True)
class BindingConstraintData:
    id: str
    field: str
    operation: Optional[Operation] = None
    timeseries_file_type: Optional[str] = None


@dataclass(frozen=True)
class ThermalData:
    area: str
    cluster: str
    column: Optional[int] = None
    field: Optional[str] = None
    operation: Optional[Operation] = None
    timeseries_file_type: Optional[str] = None


@dataclass(frozen=True)
class LinkData:
    column: int
    area_from: str
    area_to: str
    timeseries_file_type: str
    operation: Optional[Operation] = None
