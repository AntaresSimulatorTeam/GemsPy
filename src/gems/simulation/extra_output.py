from dataclasses import dataclass, field
from typing import Any, Dict

from gems.expression import evaluate
from gems.expression.evaluate import EvaluationError, ValueProvider
from gems.simulation.optimization import OptimizationProblem
from gems.study.data import ComponentParameterIndex, TimeScenarioIndex


@dataclass
class ExtraOutput:
    """Stores evaluated outputs (from ExpressionNodes), not solver variables."""

    name: str
    values: Dict[TimeScenarioIndex, float] = field(default_factory=dict)

    def set(self, t: int, s: int, value: float) -> None:
        self.values[TimeScenarioIndex(t, s)] = value

    def get(self, t: int, s: int) -> float | None:
        return self.values.get(TimeScenarioIndex(t, s))

    def is_close(
        self, other: "ExtraOutput", *, rel_tol: float = 1e-9, abs_tol: float = 0.0
    ) -> bool:
        if self.name != other.name:
            return False
        if self.values.keys() != other.values.keys():
            return False
        import math

        return all(
            math.isclose(
                self.values[k], other.values[k], rel_tol=rel_tol, abs_tol=abs_tol
            )
            for k in self.values
        )


# Component extra output evaluation
def evaluate_extra_outputs_for_a_component(
    component: Any, problem: OptimizationProblem | None
) -> Dict[str, ExtraOutput]:
    """Evaluate model-defined extra outputs for a single component."""
    results: Dict[str, ExtraOutput] = {}

    model = getattr(component, "model", None)
    if model and hasattr(model, "extra_outputs"):
        outputs = model.extra_outputs or {}
    else:
        outputs = {}

    if problem is None:
        raise ValueError("Expected a valid OptimizationProblem, got None.")

    results: Dict[str, ExtraOutput] = {}

    if not outputs:
        return results

    # Collect all time/scenario indices from the component’s variables
    all_indices = set()
    if hasattr(component, "_variables"):
        for var in component._variables.values():
            if hasattr(var, "_value"):
                all_indices.update(var._value.keys())

    if not all_indices:
        all_indices = {TimeScenarioIndex(0, 0)}

    sorted_indices = sorted(all_indices, key=lambda k: (k.time, k.scenario))

    for idx in sorted_indices:
        for out_id, expr_node in outputs.items():
            try:
                expanded_expr = problem.context.expand_operators(expr_node)
                provider = ExtraOutputValueProvider(component, problem, idx)
                val = float(evaluate(expanded_expr, provider))
            except EvaluationError as e:
                print(
                    f"[ERROR] Eval failed for '{out_id}' in {component._id} at t={idx.time}, s={idx.scenario}: {e}"
                )
                val = float("nan")
            except Exception as e:
                print(
                    f"[ERROR] Unexpected error for '{out_id}' in {component._id}: {e}"
                )
                val = float("nan")

            if out_id not in results:
                results[out_id] = ExtraOutput(out_id)
            results[out_id].set(idx.time, idx.scenario, val)

    return results

    # Value provider


class ExtraOutputValueProvider(ValueProvider):
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
