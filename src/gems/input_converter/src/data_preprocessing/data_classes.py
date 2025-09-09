from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field


def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    class Config:
        alias_generator = _to_kebab
        extra = "forbid"
        populate_by_name = True


@dataclass(frozen=True)
class Operation:
    type: Optional[str] = None
    multiply_by: Optional[Union[str, float]] = None
    divide_by: Optional[Union[str, float]] = None

    def execute(
        self,
        initial_value: Union[pd.DataFrame, pd.Series, float],
        preprocessed_values: Optional[Union[dict[str, float], float]] = None,
    ) -> Union[float, pd.Series, pd.DataFrame]:
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


class ObjectProperties(ModifiedBaseModel):
    type: str
    area: Optional[str] = None
    binding_constraint_id: Optional[str] = Field(None, alias="binding-constraint-id")
    cluster: Optional[str] = None
    link: Optional[str] = None
    field: Optional[str] = None


class ComplexData(ModifiedBaseModel):
    object_properties: Optional[ObjectProperties] = Field(
        None, alias="object-properties"
    )
    operation: Optional[Operation] = None
    column: Optional[int] = None
