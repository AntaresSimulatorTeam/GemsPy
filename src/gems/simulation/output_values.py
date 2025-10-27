# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""
Utility classes to obtain solver results.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeVar, Union, cast

from gems.simulation.extra_output import ExtraOutput
from gems.simulation.optimization import OptimizationProblem
from gems.study.data import TimeScenarioIndex


@dataclass
class OutputValues:
    """
    Contains variables and extra outputs after solver work completion.
    """

    @dataclass
    class Variable:
        _name: str
        _value: Dict[TimeScenarioIndex, float] = field(init=False, default_factory=dict)
        _size: Tuple[int, int] = field(init=False, default=(0, 0))
        _basis_status: Dict[TimeScenarioIndex, str] = field(
            init=False, default_factory=dict
        )
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
                    return self._value[TimeScenarioIndex(0, 0)]
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
            if isinstance(values, list):
                size_s = len(values)
                for scenario, timesteps in enumerate(values):
                    if isinstance(timesteps, list):
                        size_t = len(timesteps)
                        for timestep, value in enumerate(timesteps):
                            self._value[TimeScenarioIndex(timestep, scenario)] = value
                    else:
                        self._value[TimeScenarioIndex(0, scenario)] = cast(
                            float, timesteps
                        )
            else:
                self._value[TimeScenarioIndex(0, 0)] = values
            self._size = (size_s, size_t)

        def _set(
            self,
            timestep: Optional[int],
            scenario: Optional[int],
            value: float,
            status: Optional[str] = None,
            is_mip: bool = True,
        ) -> None:
            timestep = 0 if timestep is None else timestep
            scenario = 0 if scenario is None else scenario
            key = TimeScenarioIndex(timestep, scenario)
            if key not in self._value:
                size_s = max(self._size[0], scenario + 1)
                size_t = max(self._size[1], timestep + 1)
                self._size = (size_s, size_t)

            self._value[key] = value
            if not is_mip and status is not None:
                self._basis_status[key] = status

    @dataclass
    class Component:
        _id: str
        _variables: Dict[str, "OutputValues.Variable"] = field(
            init=False, default_factory=dict
        )
        _extra_outputs: Dict[str, ExtraOutput] = field(init=False, default_factory=dict)
        model: Optional[Any] = field(default=None, init=False)
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
                and _are_mappings_close(
                    self._extra_outputs, other._extra_outputs, rel_tol, abs_tol
                )
            )

        def __str__(self) -> str:
            string = f"{self._id} : {'(ignored)' if self.ignore else ''}\n"
            for var in self._variables.values():
                string += f"  {str(var)}\n"
            if self._extra_outputs:
                string += "  [Extra Outputs]\n"
                for out in self._extra_outputs.values():
                    string += f"    {out.name}: {out.values}\n"
            return string

        def var(self, variable_name: str) -> "OutputValues.Variable":
            if variable_name not in self._variables:
                self._variables[variable_name] = OutputValues.Variable(variable_name)
            return self._variables[variable_name]

        def evaluate_extra_outputs(self, problem: Any) -> None:
            """Evaluate this component’s model-defined extra outputs."""
            from gems.simulation.extra_output import (
                evaluate_extra_outputs_for_a_component,
            )

            self._extra_outputs.clear()
            self._extra_outputs.update(
                evaluate_extra_outputs_for_a_component(self, problem)
            )

    problem: Optional[OptimizationProblem] = field(default=None)
    _components: Dict[str, "OutputValues.Component"] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._build_components()
        self.evaluate_extra_outputs()

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
        return "\n" + "".join(f"{comp}\n" for comp in self._components.values())

    def _build_components(self) -> None:
        if self.problem is None:
            return

        is_mip = self.problem.solver.IsMip()
        for key, value in self.problem.context.get_all_component_variables().items():
            status = None if is_mip else value.basis_status()
            self.component(key.component_id).var(str(key.variable_name))._set(
                key.block_timestep,
                key.scenario,
                value.solution_value(),
                status=status,
                is_mip=is_mip,
            )

        for cmp in self.problem.context.network.all_components:
            comp = self.component(cmp.id)
            comp.model = cmp.model

    def component(self, component_id: str) -> "OutputValues.Component":
        if component_id not in self._components:
            self._components[component_id] = OutputValues.Component(component_id)
        return self._components[component_id]

    def evaluate_extra_outputs(self) -> None:
        """Evaluate extra outputs for all components."""
        for comp in self._components.values():
            comp.evaluate_extra_outputs(self.problem)


Comparable = TypeVar(
    "Comparable", OutputValues.Component, OutputValues.Variable, ExtraOutput
)


def _are_mappings_close(
    lhs: Mapping[str, Comparable],
    rhs: Mapping[str, Comparable],
    rel_tol: float,
    abs_tol: float,
) -> bool:
    lhs_keys = lhs.keys()
    rhs_keys = rhs.keys()

    # Keys present only on the left
    lhs_only = lhs_keys - rhs_keys
    if lhs_only:
        for key in lhs_only:
            item = lhs[key]
            if getattr(item, "ignore", False) is False:
                return False

    # Keys present only on the right
    rhs_only = rhs_keys - lhs_keys
    if rhs_only:
        for key in rhs_only:
            item = rhs[key]
            if getattr(item, "ignore", False) is False:
                return False

    # Keys in common
    for key in lhs_keys & rhs_keys:
        left_item = lhs[key]
        right_item = rhs[key]
        if hasattr(left_item, "is_close"):
            if not left_item.is_close(right_item, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        else:
            if left_item != right_item:
                return False

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
