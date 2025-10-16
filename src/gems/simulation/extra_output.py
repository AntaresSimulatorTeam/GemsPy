from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from gems.expression import evaluate
from gems.expression.evaluate import EvaluationError, ValueProvider
from gems.expression.expression import ExpressionNode
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


def evaluate_all_extra_outputs(
    problem: Any, component_lookup: Callable[[str], Any]
) -> Dict[str, Dict[str, ExtraOutput]]:
    """
    Evaluate all model-defined extra outputs for every component in the network.
    """
    if problem is None or getattr(problem, "context", None) is None:
        return {}

    database = getattr(problem.context, "database", None)
    if database is None:
        print("[WARN] No database found in problem context; extra outputs skipped.")
        return {}

    results: Dict[str, Dict[str, ExtraOutput]] = {}

    for cmp in getattr(problem.context.network, "all_components", []):
        model_def = getattr(cmp, "model", None)
        if not getattr(model_def, "extra_outputs", None):
            continue

        comp_obj = component_lookup(getattr(cmp, "id"))
        extra_results = evaluate_extra_outputs_for_a_component(comp_obj, problem)

        if extra_results:
            results[getattr(cmp, "id")] = extra_results

    return results


def evaluate_extra_outputs_for_a_component(
    component: Any, problem: Any
) -> Dict[str, ExtraOutput]:
    """
    Evaluate one component's model-defined extra outputs using solver variables as context.
    """
    results: Dict[str, ExtraOutput] = {}
    model_def = getattr(component, "model", None)
    outputs: Dict[str, ExpressionNode] = getattr(model_def, "extra_outputs", {}) or {}
    if not outputs:
        return results

    # Determine all time/scenario indices
    all_indices: set[TimeScenarioIndex] = {
        idx
        for var in getattr(component, "_variables", {}).values()
        for idx in getattr(var, "_value", {})
    }
    if not all_indices:
        all_indices = {TimeScenarioIndex(0, 0)}
    sorted_indices = sorted(all_indices, key=lambda k: (k.time, k.scenario))

    # Evaluate all ExpressionNodes for each index
    for idx in sorted_indices:
        for out_id, expr_node in outputs.items():
            try:
                expanded_expr = problem.context.expand_operators(expr_node)
                value_provider = ExtraOutputValueProvider(component, problem, idx)
                val: float = float(evaluate(expanded_expr, value_provider))
            except EvaluationError as e:
                print(
                    f"[ERROR] Failed to evaluate extra output '{out_id}' "
                    f"for {getattr(component, '_id', '')} at t={idx.time}, s={idx.scenario}: {e}"
                )
                val = float("nan")
            except Exception as e:
                print(
                    f"[ERROR] Unexpected error evaluating '{out_id}' "
                    f"for {getattr(component, '_id', '')} at t={idx.time}, s={idx.scenario}: {e}"
                )
                val = float("nan")

            if out_id not in results:
                results[out_id] = ExtraOutput(out_id)
            results[out_id].set(idx.time, idx.scenario, val)

    return results


class ExtraOutputValueProvider(ValueProvider):
    """
    ValueProvider that builds context from a component, problem, and time/scenario index.
    """

    component: Any
    problem: Any
    idx: TimeScenarioIndex
    context: Dict[str, float]

    def __init__(self, component: Any, problem: Any, idx: TimeScenarioIndex) -> None:
        self.component = component
        self.problem = problem
        self.idx = idx
        self.context = self._build_context()

    def _build_context(self) -> Dict[str, float]:
        ctx: Dict[str, float] = {}

        # Add variables with both direct and component-qualified names
        for vname, vobj in getattr(self.component, "_variables", {}).items():
            val = getattr(vobj, "_value", {}).get(self.idx)
            if val is not None:
                ctx[vname] = val
                ctx[f"{getattr(self.component, '_id', '')}.{vname}"] = val

        # Add parameters with both direct and component-qualified names
        param_names = getattr(getattr(self.component, "model", {}), "parameters", [])
        for pname in param_names:
            try:
                val = self.problem.context.database.get_value(
                    ComponentParameterIndex(getattr(self.component, "_id", ""), pname),
                    self.idx.time,
                    self.idx.scenario,
                )
                ctx[pname] = val
                ctx[f"{getattr(self.component, '_id', '')}.{pname}"] = val
            except KeyError:
                continue

        return ctx

    def get_variable_value(self, name: str) -> float:
        return self.context[name]

    def get_parameter_value(self, name: str) -> float:
        return self.context[name]

    def get_component_variable_value(self, component_id: str, name: str) -> float:
        return self.context[name]

    def get_component_parameter_value(self, component_id: str, name: str) -> float:
        return self.context[name]
