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
from dataclasses import dataclass, field
from typing import Set

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorStrategy import BailErrorStrategy

from gems_craft.expression import ExpressionNode, literal, param, relative_time, var
from gems_craft.expression.equality import expressions_equal
from gems_craft.expression.expression import (
    Comparator,
    ComparisonNode,
    DualNode,
    LowerBoundNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ReducedCostNode,
    UpperBoundNode,
    maximum,
    minimum,
)
from gems_craft.expression.parsing.antlr.ExprLexer import ExprLexer
from gems_craft.expression.parsing.antlr.ExprParser import ExprParser
from gems_craft.expression.parsing.antlr.ExprVisitor import ExprVisitor


@dataclass(frozen=True)
class ModelIdentifiers:
    """
    Allows to distinguish between parameters, variables, and constraints.
    """

    variables: Set[str]
    parameters: Set[str]
    constraints: Set[str] = field(default_factory=set)

    def is_variable(self, identifier: str) -> bool:
        return identifier in self.variables

    def is_parameter(self, identifier: str) -> bool:
        return identifier in self.parameters


@dataclass(frozen=True)
class ExpressionNodeBuilderVisitor(ExprVisitor):
    """
    Visits a tree created by ANTLR to create our AST representation.
    """

    identifiers: ModelIdentifiers

    # Visit a parse tree produced by ExprParser#portFieldExpr.
    def visitPortFieldExpr(
        self, ctx: ExprParser.PortFieldExprContext
    ) -> ExpressionNode:
        return PortFieldNode(
            port_name=ctx.IDENTIFIER(0).getText(),  # type: ignore
            field_name=ctx.IDENTIFIER(1).getText(),  # type: ignore
        )

    def visitFullexpr(self, ctx: ExprParser.FullexprContext) -> ExpressionNode:
        return ctx.expr().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#number.
    def visitNumber(self, ctx: ExprParser.NumberContext) -> ExpressionNode:
        return literal(float(ctx.NUMBER().getText()))  # type: ignore

    # Visit a parse tree produced by ExprParser#identifier.
    def visitIdentifier(self, ctx: ExprParser.IdentifierContext) -> ExpressionNode:
        return self._convert_identifier(ctx.IDENTIFIER().getText())  # type: ignore

    # Visit a parse tree produced by ExprParser#division.
    def visitMuldiv(self, ctx: ExprParser.MuldivContext) -> ExpressionNode:
        left = ctx.expr(0).accept(self)  # type: ignore
        right = ctx.expr(1).accept(self)  # type: ignore
        op = ctx.op.text  # type: ignore
        if op == "*":
            return left * right
        elif op == "/":
            return left / right
        raise ValueError(f"Invalid operator {op}")

    # Visit a parse tree produced by ExprParser#subtraction.
    def visitAddsub(self, ctx: ExprParser.AddsubContext) -> ExpressionNode:
        left = ctx.expr(0).accept(self)  # type: ignore
        right = ctx.expr(1).accept(self)  # type: ignore
        op = ctx.op.text  # type: ignore
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        raise ValueError(f"Invalid operator {op}")

    # Visit a parse tree produced by ExprParser#negation.
    def visitNegation(self, ctx: ExprParser.NegationContext) -> ExpressionNode:
        return -ctx.expr().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#expression.
    def visitExpression(self, ctx: ExprParser.ExpressionContext) -> ExpressionNode:
        return ctx.expr().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#unsignedAtom.
    def visitUnsignedAtom(self, ctx: ExprParser.UnsignedAtomContext) -> ExpressionNode:
        return ctx.atom().accept(self)  # type: ignore

    def _convert_identifier(self, identifier: str) -> ExpressionNode:
        if self.identifiers.is_variable(identifier):
            return var(identifier)
        elif self.identifiers.is_parameter(identifier):
            return param(identifier)
        raise ValueError(f"{identifier} is not a valid variable or parameter name.")

    # Visit a parse tree produced by ExprParser#comparison.
    def visitComparison(self, ctx: ExprParser.ComparisonContext) -> ExpressionNode:
        op = ctx.COMPARISON().getText()  # type: ignore
        exp1 = ctx.expr(0).accept(self)  # type: ignore
        exp2 = ctx.expr(1).accept(self)  # type: ignore
        comp = {
            "=": Comparator.EQUAL,
            "<=": Comparator.LESS_THAN,
            ">=": Comparator.GREATER_THAN,
        }[op]
        return ComparisonNode(exp1, exp2, comp)

    # Visit a parse tree produced by ExprParser#portFieldSum.
    def visitPortFieldSum(self, ctx: ExprParser.PortFieldSumContext) -> ExpressionNode:
        return PortFieldAggregatorNode(ctx.portFieldExpr().accept(self), "PortSum")  # type: ignore

    # Visit a parse tree produced by ExprParser#timeShift.
    def visitTimeIndex(self, ctx: ExprParser.TimeIndexContext) -> ExpressionNode:
        expr = self._convert_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        eval_time = ctx.time_expr().accept(self)  # type: ignore
        return expr.eval(eval_time)

    def visitTimeShift(self, ctx: ExprParser.TimeShiftContext) -> ExpressionNode:
        shifted_expr = self._convert_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        return _shift(shifted_expr, ctx.shift().accept(self))  # type: ignore

    def visitTimeShiftExpr(
        self, ctx: ExprParser.TimeShiftExprContext
    ) -> ExpressionNode:
        return _shift(ctx.expr().accept(self), ctx.shift().accept(self))  # type: ignore

    def visitTimeIndexExpr(
        self, ctx: ExprParser.TimeIndexExprContext
    ) -> ExpressionNode:
        expr = ctx.expr().accept(self)  # type: ignore
        eval_time = ctx.time_expr().accept(self)  # type: ignore
        return expr.eval(eval_time)

    def visitTimeSum(self, ctx: ExprParser.TimeSumContext) -> ExpressionNode:
        summed_expr = ctx.expr().accept(self)  # type: ignore
        from_bound = ctx.from_.accept(self)  # type: ignore
        to_bound = ctx.to.accept(self)  # type: ignore
        return summed_expr.time_sum_between(from_bound, to_bound)

    def visitSum_bound(self, ctx: ExprParser.Sum_boundContext) -> ExpressionNode:
        """A shift ("t", "t + ...") is relative to the current time step; any
        other expression is an absolute time index."""
        if ctx.shift() is not None:  # type: ignore
            return relative_time(ctx.shift().accept(self))  # type: ignore
        return ctx.time_expr().accept(self)  # type: ignore

    def visitAllTimeSum(self, ctx: ExprParser.AllTimeSumContext) -> ExpressionNode:
        shifted_expr = ctx.expr().accept(self)  # type: ignore
        return shifted_expr.time_sum()

    def _visit_dual(self, arg_exprs: list) -> ExpressionNode:
        if len(arg_exprs) != 1:
            raise ValueError("dual() requires exactly 1 argument.")
        cid = arg_exprs[0].getText()  # type: ignore
        if cid not in self.identifiers.constraints:
            raise ValueError(f"'{cid}' is not a constraint of the model.")
        return DualNode(cid)

    def _visit_reduced_cost(self, arg_exprs: list) -> ExpressionNode:
        if len(arg_exprs) != 1:
            raise ValueError("reduced_cost() requires exactly 1 argument.")
        vid = arg_exprs[0].getText()  # type: ignore
        if vid not in self.identifiers.variables:
            raise ValueError(f"'{vid}' is not a variable of the model.")
        return ReducedCostNode(vid)

    def _visit_lower_bound(self, arg_exprs: list) -> ExpressionNode:
        if len(arg_exprs) != 1:
            raise ValueError("lower_bound() requires exactly 1 argument.")
        vid = arg_exprs[0].getText()  # type: ignore
        if vid not in self.identifiers.variables:
            raise ValueError(f"'{vid}' is not a variable of the model.")
        return LowerBoundNode(vid)

    def _visit_upper_bound(self, arg_exprs: list) -> ExpressionNode:
        if len(arg_exprs) != 1:
            raise ValueError("upper_bound() requires exactly 1 argument.")
        vid = arg_exprs[0].getText()  # type: ignore
        if vid not in self.identifiers.variables:
            raise ValueError(f"'{vid}' is not a variable of the model.")
        return UpperBoundNode(vid)

    # Visit a parse tree produced by ExprParser#function.
    def visitFunction(self, ctx: ExprParser.FunctionContext) -> ExpressionNode:
        function_name: str = ctx.IDENTIFIER().getText()  # type: ignore
        arg_list = ctx.argList()  # type: ignore
        arg_exprs = arg_list.expr() if arg_list is not None else []  # type: ignore

        if function_name == "dual":
            return self._visit_dual(arg_exprs)
        if function_name == "reduced_cost":
            return self._visit_reduced_cost(arg_exprs)
        if function_name == "lower_bound":
            return self._visit_lower_bound(arg_exprs)
        if function_name == "upper_bound":
            return self._visit_upper_bound(arg_exprs)

        args = [expr.accept(self) for expr in arg_exprs]  # type: ignore
        return _apply_function(function_name, args)

    # Visit a parse tree produced by ExprParser#shift.
    def visitShift(self, ctx: ExprParser.ShiftContext) -> ExpressionNode:
        """Offset of "t + ...": the terms folded from left to right."""
        terms = [term.accept(self) for term in ctx.terms]  # type: ignore
        if not terms:
            return literal(0)
        return _fold(-terms[0] if ctx.ops[0].text == "-" else terms[0], ctx.ops[1:], terms[1:])  # type: ignore

    # Visit a parse tree produced by ExprParser#time_expr.
    def visitTime_expr(self, ctx: ExprParser.Time_exprContext) -> ExpressionNode:
        terms = [term.accept(self) for term in ctx.time_term()]  # type: ignore
        first = -terms[0] if ctx.neg is not None else terms[0]  # type: ignore
        return _fold(first, ctx.ops, terms[1:])  # type: ignore

    # Visit a parse tree produced by ExprParser#time_term.
    def visitTime_term(self, ctx: ExprParser.Time_termContext) -> ExpressionNode:
        factors = [factor.accept(self) for factor in ctx.time_factor()]  # type: ignore
        return _fold(factors[0], ctx.ops, factors[1:])  # type: ignore

    # Visit a parse tree produced by ExprParser#timeNumber.
    def visitTimeNumber(self, ctx: ExprParser.TimeNumberContext) -> ExpressionNode:
        return literal(float(ctx.NUMBER().getText()))  # type: ignore

    # Visit a parse tree produced by ExprParser#timeIdentifier.
    def visitTimeIdentifier(
        self, ctx: ExprParser.TimeIdentifierContext
    ) -> ExpressionNode:
        return self._convert_time_identifier(ctx.IDENTIFIER().getText())  # type: ignore

    # Visit a parse tree produced by ExprParser#timeFactorIndex.
    def visitTimeFactorIndex(
        self, ctx: ExprParser.TimeFactorIndexContext
    ) -> ExpressionNode:
        expr = self._convert_time_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        return expr.eval(ctx.time_expr().accept(self))  # type: ignore

    # Visit a parse tree produced by ExprParser#timeFactorShift.
    def visitTimeFactorShift(
        self, ctx: ExprParser.TimeFactorShiftContext
    ) -> ExpressionNode:
        expr = self._convert_time_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        return _shift(expr, ctx.shift().accept(self))  # type: ignore

    def _convert_time_identifier(self, identifier: str) -> ExpressionNode:
        if self.identifiers.is_variable(identifier):
            raise ValueError(
                f"Variable '{identifier}' is not allowed in a time shift, time index "
                "or time sum bound: only parameters and literals are."
            )
        return self._convert_identifier(identifier)

    # Visit a parse tree produced by ExprParser#timeFunction.
    def visitTimeFunction(self, ctx: ExprParser.TimeFunctionContext) -> ExpressionNode:
        function_name: str = ctx.IDENTIFIER().getText()  # type: ignore
        if (
            function_name not in _UNARY_FUNCTIONS
            and function_name not in _N_ARY_FUNCTIONS
        ):
            raise ValueError(
                f"Function {function_name} is not allowed in a time shift, time index "
                "or time sum bound."
            )
        args = [expr.accept(self) for expr in ctx.time_expr()]  # type: ignore
        return _apply_function(function_name, args)

    # Visit a parse tree produced by ExprParser#timeParenthesis.
    def visitTimeParenthesis(
        self, ctx: ExprParser.TimeParenthesisContext
    ) -> ExpressionNode:
        return ctx.time_expr().accept(self)  # type: ignore


def _shift(expr: ExpressionNode, time_shift: ExpressionNode) -> ExpressionNode:
    """expr shifted by time_shift; x[t] is x itself."""
    if expressions_equal(time_shift, literal(0)):
        return expr
    return expr.shift(time_shift)


def _fold(
    first: ExpressionNode, ops: list, operands: list[ExpressionNode]
) -> ExpressionNode:
    """Applies the binary operator tokens ops from left to right."""
    result = first
    for op, operand in zip(ops, operands):
        result = _BINARY_OPERATORS[op.text](result, operand)
    return result


_BINARY_OPERATORS = {
    "+": ExpressionNode.__add__,
    "-": ExpressionNode.__sub__,
    "*": ExpressionNode.__mul__,
    "/": ExpressionNode.__truediv__,
}


_UNARY_FUNCTIONS = {
    "expec": ExpressionNode.expec,
    "floor": ExpressionNode.floor,
    "ceil": ExpressionNode.ceil,
    "abs": ExpressionNode.abs,
    "round": ExpressionNode.round,
}

_N_ARY_FUNCTIONS = {
    "max": maximum,
    "min": minimum,
}


def _apply_function(name: str, args: list[ExpressionNode]) -> ExpressionNode:
    if name in _UNARY_FUNCTIONS:
        if len(args) != 1:
            raise ValueError(
                f"Function {name} requires exactly 1 argument, got {len(args)}"
            )
        return _UNARY_FUNCTIONS[name](args[0])
    if name in _N_ARY_FUNCTIONS:
        return _N_ARY_FUNCTIONS[name](*args)
    raise ValueError(f"Encountered invalid function name {name}")


class ParsingException(Exception):
    pass


def parse_expression(expression: str, identifiers: ModelIdentifiers) -> ExpressionNode:
    """
    Parses a string expression to create the corresponding AST representation.
    """
    try:
        input = InputStream(expression)
        lexer = ExprLexer(input)
        stream = CommonTokenStream(lexer)
        parser = ExprParser(stream)
        parser._errHandler = BailErrorStrategy()

        return ExpressionNodeBuilderVisitor(identifiers).visit(parser.fullexpr())  # type: ignore

    except ValueError as e:
        raise ParsingException(f"An error occurred during parsing: {e}") from e
    except Exception as e:
        raise ParsingException(
            f"An error occurred during parsing: {type(e).__name__}"
        ) from e
