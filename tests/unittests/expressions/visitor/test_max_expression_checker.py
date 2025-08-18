import pytest

from gems.expression import (
    Comparator,
    ComparisonNode,
    ExpressionNode,
    VariableNode,
    max_expr,
)
from gems.expression.expression import (
    ComponentVariableNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemVariableNode,
)


@pytest.mark.parametrize(
    "node, left_arg, right_arg",
    [
        (VariableNode, ("x",), ("y",)),
        (
            PortFieldAggregatorNode,
            (ExpressionNode, "PortAggregator"),
            (ExpressionNode, "PortAggregator"),
        ),
        (PortFieldNode, ("port", "name"), ("port", "name2")),
        (
            ProblemVariableNode,
            ("comp", "name", "time_index", "scenario_index"),
            ("comp", "name", "time_index", "scenario_index"),
        ),
        (ComponentVariableNode, ("component_id", "name"), ("component_id2", "name2")),
        (ComparisonNode, ("x", "y", Comparator.GREATER_THAN), 4),
    ],
)
def test_max_expr_checker(node, left_arg, right_arg) -> None:
    left = node(*left_arg)
    right = right_arg if isinstance(right_arg, int) else node(*right_arg)
    with pytest.raises(ValueError):
        max_expr(left, right)
