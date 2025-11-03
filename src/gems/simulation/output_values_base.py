# output_values_base.py

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

from gems.study.data import TimeScenarioIndex


@dataclass
class BaseOutputValue(ABC):
    """
    Abstract Base Class providing common structure and functionality
    for Variable and ExtraOutput classes. It handles storage (_value, _size)
    and shared logic (is_close, value property).
    """

    _name: str
    _value: Dict[TimeScenarioIndex, float] = field(init=False, default_factory=dict)
    _size: Tuple[int, int] = field(init=False, default=(0, 0))
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseOutputValue):
            return NotImplemented
        return (self.ignore or other.ignore) or (
            self._name == other._name
            and self._size == other._size
            and self._value == other._value
        )

    def is_close(
        self,
        other: "BaseOutputValue",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        if not isinstance(other, BaseOutputValue):
            return NotImplemented

        return (self.ignore or other.ignore) or (
            self._name == other._name
            and self._size == other._size
            and self._value.keys() == other._value.keys()
            and all(
                math.isclose(
                    self._value[key],
                    other._value[key],
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                for key in self._value
            )
        )

    def __str__(self) -> str:
        return f"{self._name} : {str(self.value)} {'(ignored)' if self.ignore else ''}"

    @property
    def value(self) -> Union[None, float, List[float], List[List[float]]]:
        size_s, size_t = self._size
        if size_t == 1:
            if size_s == 1:
                return self._value.get(TimeScenarioIndex(0, 0))
            else:
                return [self._value[TimeScenarioIndex(0, s)] for s in range(size_s)]
        else:
            return [
                [self._value[TimeScenarioIndex(t, s)] for t in range(size_t)]
                for s in range(size_s)
            ]

    @value.setter
    def value(self, values: Union[float, List[float], List[List[float]]]) -> None:
        size_s, size_t = 1, 1
        self._value.clear()
        if isinstance(values, list):
            size_s = len(values)
            for scenario, timesteps in enumerate(values):
                if isinstance(timesteps, list):
                    size_t = len(timesteps)
                    for timestep, value in enumerate(timesteps):
                        self._value[TimeScenarioIndex(timestep, scenario)] = value
                else:
                    self._value[TimeScenarioIndex(0, scenario)] = cast(float, timesteps)
        else:
            self._value[TimeScenarioIndex(0, 0)] = values
        self._size = (size_s, size_t)

    def get(self, t: int, s: int) -> float | None:
        return self._value.get(TimeScenarioIndex(t, s))
