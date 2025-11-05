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
from typing import Any, Dict, Mapping, Optional, Set, TypeVar

from gems.expression import evaluate
from gems.expression.evaluate import EvaluationError
from gems.simulation.extra_output import ExtraOutput, ExtraOutputValueProvider
from gems.simulation.optimization import OptimizationProblem
from gems.simulation.output_values_base import BaseOutputValue
from gems.study.data import TimeScenarioIndex


@dataclass
class OutputVariable(BaseOutputValue):
    """
    Contains a single solver variable's values and status.
    All shared logic is now in BaseOutputValue.
    """

    _basis_status: Dict[TimeScenarioIndex, str] = field(
        init=False, default_factory=dict
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputVariable):
            return NotImplemented
        # Check base equality first (name, size, value)
        if not super().__eq__(other):
            return False
        # Then check the unique field
        return (self.ignore or other.ignore) or (
            self._basis_status == other._basis_status
        )

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
class OutputComponent:
    _id: str
    _variables: Dict[str, OutputVariable] = field(init=False, default_factory=dict)
    _extra_outputs: Dict[str, ExtraOutput] = field(init=False, default_factory=dict)
    model: Optional[Any] = field(default=None, init=False)
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputComponent):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self,
        other: "OutputComponent",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        return (self.ignore or other.ignore) or (
            self._id == other._id
            and _are_mappings_close(self._variables, other._variables, rel_tol, abs_tol)
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
                string += f"    {out._name}: {out._value}\n"
        return string

    def var(self, variable_name: str) -> OutputVariable:
        if variable_name not in self._variables:
            self._variables[variable_name] = OutputVariable(variable_name)
        return self._variables[variable_name]

    def extra_output(self, output_name: str) -> ExtraOutput:
        if output_name not in self._extra_outputs:
            self._extra_outputs[output_name] = ExtraOutput(output_name)
        return self._extra_outputs[output_name]

    def evaluate_extra_outputs(self, problem: OptimizationProblem) -> None:
        """Evaluate all model-defined extra outputs and populate self._extra_outputs."""
        if problem is None:
            raise ValueError("Expected a valid OptimizationProblem, got None.")

        if self.model is None or self.model.extra_outputs is None:
            return

        self._extra_outputs = {}

        for out_id, expr_node in self.model.extra_outputs.items():
            if out_id not in self._extra_outputs:
                self._extra_outputs[out_id] = ExtraOutput(out_id)

            self._evaluate_single_extra_output(
                self._extra_outputs[out_id], problem, expr_node
            )

    def _evaluate_single_extra_output(
        self,
        extra_output: ExtraOutput,
        problem: OptimizationProblem,
        expr_node: Any,
    ) -> None:
        """
        Evaluate a single ExtraOutput for all time/scenario indices
        from the component's variables.
        """
        all_indices: Set[TimeScenarioIndex] = set()
        for var in self._variables.values():
            all_indices.update(var._value.keys())
        if not all_indices:
            all_indices = {TimeScenarioIndex(0, 0)}

        sorted_indices = sorted(all_indices, key=lambda k: (k.time, k.scenario))

        for idx in sorted_indices:
            try:
                expanded_expr = problem.context.expand_operators(expr_node)
                provider = ExtraOutputValueProvider(self, problem, idx)
                val = float(evaluate(expanded_expr, provider))
            except EvaluationError as e:
                print(
                    f"[ERROR] Eval failed for '{extra_output._name}' in {self._id} "
                    f"at t={idx.time}, s={idx.scenario}: {e}"
                )
                val = float("nan")
            except Exception as e:
                print(
                    f"[ERROR] Unexpected error for '{extra_output._name}' in {self._id}: {e}"
                )
                val = float("nan")

            extra_output._set(idx.time, idx.scenario, val)


@dataclass
class OutputValues:
    """
    Contains variables and extra outputs after solver work completion.
    """

    problem: Optional[OptimizationProblem] = field(default=None)
    _components: Dict[str, OutputComponent] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._build_components()
        self._fill_components()

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
        """
        Initializes component objects and links them to their models.
        It only creates the structure, no values are set.
        """
        if self.problem is None:
            return

        # Ensure a Component object exists for every component in the network
        for cmp in self.problem.context.network.all_components:
            comp = self.component(cmp.id)
            comp.model = cmp.model

    def _fill_components(self) -> None:
        """
        Fills all output values (Variables from solver, ExtraOutputs from evaluation).
        """
        # 1. Populate Variables
        self._evaluate_variables()

        # 2. Evaluate Extra Outputs, which depend on the variables being set
        self._evaluate_extra_outputs()

    def component(self, component_id: str) -> OutputComponent:
        if component_id not in self._components:
            self._components[component_id] = OutputComponent(component_id)
        return self._components[component_id]

    def _evaluate_variables(self) -> None:
        """
        Populates the OutputVariable values from the solver results. # Docstring updated
        """
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

    def _evaluate_extra_outputs(self) -> None:
        """Evaluate extra outputs for all components."""
        if self.problem is None:
            return
        for comp in self._components.values():
            comp.evaluate_extra_outputs(self.problem)


Comparable = TypeVar("Comparable", OutputComponent, OutputVariable, ExtraOutput)


def _are_mappings_close(
    lhs: Mapping[str, Comparable],
    rhs: Mapping[str, Comparable],
    rel_tol: float,
    abs_tol: float,
) -> bool:
    lhs_keys = lhs.keys()
    rhs_keys = rhs.keys()

    # Keys present only on the left
    for key in lhs_keys - rhs_keys:
        if not lhs[key].ignore:
            return False

    # Keys present only on the right
    for key in rhs_keys - lhs_keys:
        if not rhs[key].ignore:
            return False

    # Keys in common
    for key in lhs_keys & rhs_keys:
        left_item = lhs[key]
        right_item = rhs[key]
        if not left_item.ignore and not left_item.is_close(
            right_item, rel_tol=rel_tol, abs_tol=abs_tol
        ):
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
