# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""
Util class to obtain solver results
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeVar, Union, cast

from gems.simulation.optimization import OptimizationProblem
from gems.study.data import TimeScenarioIndex


@dataclass
class OutputValues:
    """
    Contents variables output values after solver work completion.
    """

    @dataclass
    class Variable:
        """
        'constant_var':      c1,
        'time_only_var':     [[t1, t2, t3]],
        'scenario_only_var': [s1, s2],
        'time_scenario_var': [[t1s1, t2s1, t3s1], [t1s2, t2s2, t3s2]]

        Internally, the _value attribute will be a dict mapping TimeScenarioIndex(t,s) to float
        """

        _name: str
        _value: Dict[TimeScenarioIndex, float] = field(init=False, default_factory=dict)
        _size: Tuple[int, int] = field(init=False, default=(0, 0))
        ignore: bool = field(default=False, init=False)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, OutputValues.Variable):
                return NotImplemented
            return (self.ignore or other.ignore) or (
                self._name == other._name
                and self._size == other._size
                and self._value == other._value
            )

        def is_close(
            self,
            other: "OutputValues.Variable",
            *,
            rel_tol: float = 1.0e-9,
            abs_tol: float = 0.0,
        ) -> bool:
            # From the docs in https://docs.python.org/3/library/math.html#math.isclose
            # math.isclose(a, b) returns abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
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
            return (
                f"{self._name} : {str(self.value)} {'(ignored)' if self.ignore else ''}"
            )

        @property
        def value(self) -> Union[None, float, List[float], List[List[float]]]:
            size_s, size_t = self._size
            if size_t == 1:
                if size_s == 1:
                    # Constant
                    return self._value[TimeScenarioIndex(0, 0)]
                else:
                    # Scenario-only
                    return [self._value[TimeScenarioIndex(0, s)] for s in range(size_s)]
            else:
                # Either Time-only or Time-Scenario
                return [
                    [self._value[TimeScenarioIndex(t, s)] for t in range(size_t)]
                    for s in range(size_s)
                ]

        @value.setter
        def value(self, values: Union[float, List[float], List[List[float]]]) -> None:
            size_s, size_t = 1, 1

            if isinstance(values, list):
                size_s = len(values)
                for scenario, timesteps in enumerate(values):
                    if isinstance(timesteps, list):
                        size_t = len(timesteps)
                        for timestep, value in enumerate(timesteps):
                            # Either Time-only or Time-Scenario
                            self._value[TimeScenarioIndex(timestep, scenario)] = value
                    else:
                        # Scenario-only
                        self._value[TimeScenarioIndex(0, scenario)] = cast(
                            float, timesteps
                        )
            else:
                # Constant
                self._value[TimeScenarioIndex(0, 0)] = values

            self._size = (size_s, size_t)

        def _set(
            self, timestep: Optional[int], scenario: Optional[int], value: float
        ) -> None:
            timestep = 0 if timestep is None else timestep
            scenario = 0 if scenario is None else scenario
            key = TimeScenarioIndex(timestep, scenario)
            if key not in self._value:
                size_s = max(self._size[0], scenario + 1)
                size_t = max(self._size[1], timestep + 1)
                self._size = (size_s, size_t)

            self._value[key] = value

    @dataclass
    class Component:
        _id: str
        _variables: Dict[str, "OutputValues.Variable"] = field(
            init=False, default_factory=dict
        )
        ignore: bool = field(default=False, init=False)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, OutputValues.Component):
                return NotImplemented
            return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

        def is_close(
            self,
            other: "OutputValues.Component",
            *,
            rel_tol: float = 1.0e-9,
            abs_tol: float = 0.0,
        ) -> bool:
            return (self.ignore or other.ignore) or (
                self._id == other._id
                and _are_mappings_close(
                    self._variables, other._variables, rel_tol, abs_tol
                )
            )

        def __str__(self) -> str:
            string = f"{self._id} : {'(ignored)' if self.ignore else ''}\n"
            for var in self._variables.values():
                string += f"  {str(var)}\n"
            return string

        def var(self, variable_name: str) -> "OutputValues.Variable":
            if variable_name not in self._variables:
                self._variables[variable_name] = OutputValues.Variable(variable_name)
            return self._variables[variable_name]

    problem: Optional[OptimizationProblem] = field(default=None)
    _components: Dict[str, "OutputValues.Component"] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._build_components()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputValues):
            return NotImplemented
        return _are_mappings_close(self._components, other._components, 0.0, 0.0)

    def is_close(
        self, other: "OutputValues", *, rel_tol: float = 1.0e-9, abs_tol: float = 0.0
    ) -> bool:
        return _are_mappings_close(
            self._components, other._components, rel_tol, abs_tol
        )

    def __str__(self) -> str:
        string = "\n"
        for comp in self._components.values():
            string += f"{str(comp)}"

        return string

    def _build_components(self) -> None:
        if self.problem is None:
            return

        for key, value in self.problem.context.get_all_component_variables().items():
            (
                self.component(key.component_id)
                .var(str(key.variable_name))
                ._set(key.block_timestep, key.scenario, value.solution_value())
            )

    def component(self, component_id: str) -> "OutputValues.Component":
        if component_id not in self._components:
            self._components[component_id] = OutputValues.Component(component_id)
        return self._components[component_id]


Comparable = TypeVar("Comparable", OutputValues.Component, OutputValues.Variable)


def _are_mappings_close(
    lhs: Mapping[str, Comparable],
    rhs: Mapping[str, Comparable],
    rel_tol: float,
    abs_tol: float,
) -> bool:
    lhs_keys = lhs.keys()
    rhs_keys = rhs.keys()

    if (lhs_only_keys := lhs_keys - rhs_keys) and any(
        not lhs[key].ignore for key in lhs_only_keys
    ):
        return False

    elif (rhs_only_keys := rhs_keys - lhs_keys) and any(
        not rhs[key].ignore for key in rhs_only_keys
    ):
        return False

    elif intersect_keys := lhs_keys & rhs_keys:
        if rel_tol == abs_tol == 0.0:
            return all(lhs[key] == rhs[key] for key in intersect_keys)
        else:
            return all(
                lhs[key].is_close(rhs[key], rel_tol=rel_tol, abs_tol=abs_tol)
                for key in intersect_keys
            )
    else:
        return True


@dataclass(frozen=True)
class BendersSolution:
    data: Dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BendersSolution):
            return NotImplemented
        return (
            self.overall_cost == other.overall_cost
            and self.candidates == other.candidates
        )

    def is_close(
        self,
        other: "BendersSolution",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        return (
            math.isclose(
                self.overall_cost, other.overall_cost, abs_tol=abs_tol, rel_tol=rel_tol
            )
            and self.candidates.keys() == other.candidates.keys()
            and all(
                math.isclose(
                    self.candidates[key],
                    other.candidates[key],
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                for key in self.candidates
            )
        )

    def __str__(self) -> str:
        lpad = 30
        rpad = 12

        string = "Benders' solution:\n"
        string += f"{'Overall cost':<{lpad}} : {self.overall_cost:>{rpad}}\n"
        string += f"{'Investment cost':<{lpad}} : {self.investment_cost:>{rpad}}\n"
        string += f"{'Operational cost':<{lpad}} : {self.operational_cost:>{rpad}}\n"
        string += "-" * (lpad + rpad + 3) + "\n"
        for candidate, investment in self.candidates.items():
            string += f"{candidate:<{lpad}} : {investment:>{rpad}}\n"

        return string

    @property
    def investment_cost(self) -> float:
        return self.data["solution"]["investment_cost"]

    @property
    def operational_cost(self) -> float:
        return self.data["solution"]["operational_cost"]

    @property
    def overall_cost(self) -> float:
        return self.data["solution"]["overall_cost"]

    @property
    def candidates(self) -> Dict[str, float]:
        return self.data["solution"]["values"]

    @property
    def status(self) -> str:
        return self.data["solution"]["problem_status"]

    @property
    def absolute_gap(self) -> float:
        return self.data["solution"]["optimality_gap"]

    @property
    def relative_gap(self) -> float:
        return self.data["solution"]["relative_gap"]

    @property
    def stopping_criterion(self) -> str:
        return self.data["solution"]["stopping_criterion"]


@dataclass(frozen=True, eq=False)
class BendersMergedSolution(BendersSolution):
    @property
    def lower_bound(self) -> float:
        return self.data["solution"]["lb"]

    @property
    def upper_bound(self) -> float:
        return self.data["solution"]["ub"]


@dataclass(frozen=True, eq=False)
class BendersDecomposedSolution(BendersSolution):
    @property
    def nb_iterations(self) -> int:
        return self.data["solution"]["iteration"]

    @property
    def duration(self) -> float:
        return self.data["run_duration"]
