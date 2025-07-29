from typing import Any
from gems.expression import ExpressionVisitor, visit
from gems.expression import LiteralNode, VariableNode, ParameterNode, MaxNode, TimeShiftNode
from gems.expression.visitor import visit

class _MaxExpressionChecker(ExpressionVisitor[None]):
    """
    Visits the whole expression to check there is no:
    comparison, other port field, component-associated parametrs or variables...
    """
    def literal(self, node: LiteralNode) -> None:
        pass  # Les littéraux sont constants

    def parameter(self, node: ParameterNode) -> None:
        pass  # Les paramètres sont constants

    def variable(self, node: VariableNode) -> None:
        raise ValueError("MaxNode cannot contain a variable.")

    def max_node(self, node: MaxNode) -> None:
        if len(node.operands) < 2:
            raise ValueError("MaxNode requires at least two operands")
        for operand in node.operands:
            visit(operand, self)

    def time_shift(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a time shift operation.")

    def addition(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain an addition operation.")

    def comparison(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a comparison operator.")

    def all_time_sum(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain an all-time sum operation.")

    def comp_parameter(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a component parameter.")

    def comp_variable(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a component variable.")

    def division(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a division operation.")

    def multiplication(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a multiplication operation.")

    def negation(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a negation operation.")

    def pb_parameter(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a problem parameter.")

    def pb_variable(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a problem variable.")

    def port_field(self, node: Any) -> None:
        raise ValueError("MaxNode cannot reference a port field.")

    def port_field_aggregator(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain port field aggregation.")

    def scenario_operator(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a scenario operator.")

    def time_eval(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a time evaluation operation.")

    def time_sum(self, node: Any) -> None:
        raise ValueError("MaxNode cannot contain a time sum operation.")

def _validate_max_expression(node: MaxNode) -> None:
    """
    Valide l'expression d'un MaxNode en visitant ses opérandes avec _MaxExpressionChecker.
    """
    checker = _MaxExpressionChecker()
    for operand in node.operands:
        visit(operand, checker)