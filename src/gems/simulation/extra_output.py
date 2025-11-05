from dataclasses import dataclass
from typing import Any, Dict, Optional

from gems.expression.evaluate import ValueProvider
from gems.simulation.optimization import OptimizationProblem
from gems.simulation.output_values_base import BaseOutputValue
from gems.study.data import ComponentParameterIndex, TimeScenarioIndex


@dataclass
class ExtraOutput(BaseOutputValue):
    """
    Stores evaluated outputs (from ExpressionNodes), not solver variables.
    Inherits all common fields (_name, _value, _size, ignore) and methods
    (__eq__, is_close, __str__, value property) from BaseOutputValue.
    """

    def _set(
        self,
        timestep: Optional[int],
        scenario: Optional[int],
        value: float,
    ) -> None:
        timestep = 0 if timestep is None else timestep
        scenario = 0 if scenario is None else scenario
        key = TimeScenarioIndex(timestep, scenario)

        if key not in self._value:
            size_s = max(self._size[0], scenario + 1)
            size_t = max(self._size[1], timestep + 1)
            self._size = (size_s, size_t)

        self._value[key] = value


class ExtraOutputValueProvider(ValueProvider):
    # ... (content remains the same, as it only interacts with public methods/inherited fields like _value) ...
    """Provides variable and parameter values for extra output expressions."""

    def __init__(
        self,
        component: Any,
        problem: OptimizationProblem,
        idx: TimeScenarioIndex,
    ) -> None:
        self.component = component
        self.problem = problem
        self.idx = idx
        self.context = self._build_context()

    def _build_context(self) -> Dict[str, float]:
        ctx: Dict[str, float] = {}

        # --- Variables ---
        if hasattr(self.component, "_variables"):
            for vname, vobj in self.component._variables.items():
                if hasattr(vobj, "_value"):
                    val = vobj._value.get(self.idx)
                    if val is not None:
                        ctx[vname] = val
                        if hasattr(self.component, "_id"):
                            ctx[f"{self.component._id}.{vname}"] = val

        # --- Parameters ---
        model = getattr(self.component, "model", None)
        if model is not None and hasattr(model, "parameters"):
            for pname in model.parameters:
                try:
                    val = self.problem.context.database.get_value(
                        ComponentParameterIndex(self.component._id, pname),
                        self.idx.time,
                        self.idx.scenario,
                    )
                    ctx[pname] = val
                    if hasattr(self.component, "_id"):
                        ctx[f"{self.component._id}.{pname}"] = val
                except KeyError:
                    continue

        return ctx

    # ValueProvider interface
    def get_variable_value(self, name: str) -> float:
        return self.context[name]

    def get_parameter_value(self, name: str) -> float:
        return self.context[name]

    def get_component_variable_value(self, component_id: str, name: str) -> float:
        return self.context[name]

    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        return self.context[name]
