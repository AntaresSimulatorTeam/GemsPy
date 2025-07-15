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
from dataclasses import dataclass
from typing import Set

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorStrategy import BailErrorStrategy

from gems.expression import ExpressionNode, literal, param, var
from gems.expression.equality import expressions_equal
from gems.expression.expression import (
    Comparator,
    ComparisonNode,
    PortFieldAggregatorNode,
    PortFieldNode,
)
from gems.expression.parsing.antlr.ExprLexer import ExprLexer
from gems.expression.parsing.antlr.ExprParser import ExprParser
from gems.expression.parsing.antlr.ExprVisitor import ExprVisitor


@dataclass(frozen=True)
class ModelIdentifiers:
    """
    Allows to distinguish between parameters and variables.
    """

    variables: Set[str]
    parameters: Set[str]

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

    # Visit a parse tree produced by ExprParser#portField.
    def visitPortField(self, ctx: ExprParser.PortFieldContext) -> ExpressionNode:
        return ctx.portFieldExpr().accept(self)  # type: ignore

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
        eval_time = ctx.expr().accept(self)  # type: ignore
        return expr.eval(eval_time)

    def visitTimeShift(self, ctx: ExprParser.TimeShiftContext) -> ExpressionNode:
        shifted_expr = self._convert_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        time_shift = ctx.shift().accept(self)  # type: ignore
        # specifics for x[t] ...
        if expressions_equal(time_shift, literal(0)):
            return shifted_expr
        return shifted_expr.shift(time_shift)

    def visitTimeSum(self, ctx: ExprParser.TimeSumContext) -> ExpressionNode:
        shifted_expr = ctx.expr().accept(self)  # type: ignore
        from_shift = ctx.from_.accept(self)  # type: ignore
        to_shift = ctx.to.accept(self)  # type: ignore
        return shifted_expr.time_sum(from_shift, to_shift)

    def visitAllTimeSum(self, ctx: ExprParser.AllTimeSumContext) -> ExpressionNode:
        shifted_expr = ctx.expr().accept(self)  # type: ignore
        return shifted_expr.time_sum()

    # Visit a parse tree produced by ExprParser#function.
    def visitFunction(self, ctx: ExprParser.FunctionContext) -> ExpressionNode:
        function_name: str = ctx.IDENTIFIER().getText()  # type: ignore
        operand: ExpressionNode = ctx.expr().accept(self)  # type: ignore
        fn = _FUNCTIONS.get(function_name, None)
        if fn is None:
            raise ValueError(f"Encountered invalid function name {function_name}")
        return fn(operand)

    # Visit a parse tree produced by ExprParser#shift.
    def visitShift(self, ctx: ExprParser.ShiftContext) -> ExpressionNode:
        if ctx.shift_expr() is None:  # type: ignore
            return literal(0)
        shift = ctx.shift_expr().accept(self)  # type: ignore
        return shift

    # Visit a parse tree produced by ExprParser#shiftAddsub.
    def visitShiftAddsub(self, ctx: ExprParser.ShiftAddsubContext) -> ExpressionNode:
        left = ctx.shift_expr().accept(self)  # type: ignore
        right = ctx.right_expr().accept(self)  # type: ignore
        op = ctx.op.text  # type: ignore
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        raise ValueError(f"Invalid operator {op}")

    # Visit a parse tree produced by ExprParser#shiftMuldiv.
    def visitShiftMuldiv(self, ctx: ExprParser.ShiftMuldivContext) -> ExpressionNode:
        left = ctx.shift_expr().accept(self)  # type: ignore
        right = ctx.right_expr().accept(self)  # type: ignore
        op = ctx.op.text  # type: ignore
        if op == "*":
            return left * right
        elif op == "/":
            return left / right
        raise ValueError(f"Invalid operator {op}")

    # Visit a parse tree produced by ExprParser#signedExpression.
    def visitSignedExpression(
        self, ctx: ExprParser.SignedExpressionContext
    ) -> ExpressionNode:
        if ctx.op.text == "-":  # type: ignore
            return -ctx.expr().accept(self)  # type: ignore
        else:
            return ctx.expr().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#signedAtom.
    def visitSignedAtom(self, ctx: ExprParser.SignedAtomContext) -> ExpressionNode:
        if ctx.op.text == "-":  # type: ignore
            return -ctx.atom().accept(self)  # type: ignore
        else:
            return ctx.atom().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#rightExpression.
    def visitRightExpression(
        self, ctx: ExprParser.RightExpressionContext
    ) -> ExpressionNode:
        return ctx.expr().accept(self)  # type: ignore

    # Visit a parse tree produced by ExprParser#rightMuldiv.
    def visitRightMuldiv(self, ctx: ExprParser.RightMuldivContext) -> ExpressionNode:
        left = ctx.right_expr(0).accept(self)  # type: ignore
        right = ctx.right_expr(1).accept(self)  # type: ignore
        op = ctx.op.text  # type: ignore
        if op == "*":
            return left * right
        elif op == "/":
            return left / right
        raise ValueError(f"Invalid operator {op}")

    # Visit a parse tree produced by ExprParser#rightAtom.
    def visitRightAtom(self, ctx: ExprParser.RightAtomContext) -> ExpressionNode:
        return ctx.atom().accept(self)  # type: ignore


_FUNCTIONS = {
    "expec": ExpressionNode.expec,
}


class AntaresParseException(Exception):
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
        raise AntaresParseException(f"An error occurred during parsing: {e}") from e
    except Exception as e:
        raise AntaresParseException(
            f"An error occurred during parsing: {type(e).__name__}"
        ) from e
