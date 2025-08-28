from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class ConversionMode(Enum):
    HYBRID = "hybrid"
    FULL = "full"


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
class ObjectProperties:
    type: str
    area: Optional[str] = None
    binding_constraint_id: Optional[str] = None
    cluster: Optional[str] = None
    link: Optional[str] = None
    field: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_data: dict):
        yaml_data = yaml_data.copy()
        if "binding-constraint-id" in yaml_data:
            yaml_data["binding_constraint_id"] = yaml_data.pop("binding-constraint-id")
        return cls(**yaml_data)


@dataclass(frozen=True)
class MatrixData:
    object_properties: Optional[ObjectProperties] = None

    @classmethod
    def from_yaml(cls, yaml_data: dict):
        yaml_data = yaml_data.copy()
        if "object-properties" in yaml_data:
            yaml_data["object_properties"] = yaml_data.pop("object-properties")
        return cls(**yaml_data)


@dataclass(frozen=True)
class ComplexData:
    object_properties: Optional[ObjectProperties] = None
    operation: Optional[Operation] = None
    column: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_data: dict):
        yaml_data = yaml_data.copy()
        if "object-properties" in yaml_data:
            yaml_data["object_properties"] = yaml_data.pop("object-properties")
        return cls(**yaml_data)
