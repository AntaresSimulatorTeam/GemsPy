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
from typing import Callable, Dict, Optional, Set, Tuple

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorStrategy import BailErrorStrategy

from gems_craft.expression import ExpressionNode, literal, param, var
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
    sets: Set[str] = field(default_factory=set)

    def is_variable(self, identifier: str) -> bool:
        return identifier in self.variables

    def is_parameter(self, identifier: str) -> bool:
        return identifier in self.parameters

    def is_set(self, identifier: str) -> bool:
        return identifier in self.sets


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

    # Visit a parse tree produced by ExprParser#bracketIndex.
    def visitBracketIndex(self, ctx: ExprParser.BracketIndexContext) -> ExpressionNode:
        base = self._convert_identifier(ctx.IDENTIFIER().getText())  # type: ignore
        return self._apply_index_list(base, ctx.indexList())  # type: ignore

    def visitBracketIndexExpr(
        self, ctx: ExprParser.BracketIndexExprContext
    ) -> ExpressionNode:
        base = ctx.expr().accept(self)  # type: ignore
        return self._apply_index_list(base, ctx.indexList())  # type: ignore

    def visitSumOver(self, ctx: ExprParser.SumOverContext) -> ExpressionNode:
        set_id: str = ctx.IDENTIFIER().getText()  # type: ignore
        if not self.identifiers.is_set(set_id):
            raise ValueError(
                f"'{set_id}' is not a declared set; sum_over() requires a declared set id."
            )
        operand = ctx.expr().accept(self)  # type: ignore
        return operand.sum_over(set_id)

    def _apply_index_list(
        self, base: ExpressionNode, index_list_ctx: "ExprParser.IndexListContext"
    ) -> ExpressionNode:
        """
        Applies every term of an index list (custom sets and indexing
        extension) to `base`. Term order in the *source syntax* must not
        affect the resulting AST (`X[fuel=3, 2]` == `X[2, fuel=3]`), so terms
        are first classified independently of one another, then applied in a
        canonical order: the (at most one) implicit time term first, then set
        terms sorted by set id.
        """
        time_applier: Optional[Callable[[ExpressionNode], ExpressionNode]] = None
        set_appliers: Dict[str, Callable[[ExpressionNode], ExpressionNode]] = {}

        for term_ctx in index_list_ctx.indexTerm():  # type: ignore
            set_id, apply = term_ctx.accept(self)  # type: ignore
            if set_id is None:
                if time_applier is not None:
                    raise ValueError(
                        "An index list cannot contain more than one term denoting "
                        "the (implicit) time dimension."
                    )
                time_applier = apply
            else:
                if set_id in set_appliers:
                    raise ValueError(f"Set '{set_id}' is indexed more than once.")
                set_appliers[set_id] = apply

        result = base
        if time_applier is not None:
            result = time_applier(result)
        for set_id in sorted(set_appliers):
            result = set_appliers[set_id](result)
        return result

    # Visit a parse tree produced by ExprParser#namedOrTimeShiftTerm.
    def visitNamedOrTimeShiftTerm(
        self, ctx: ExprParser.NamedOrTimeShiftTermContext
    ) -> Tuple[Optional[str], Callable[[ExpressionNode], ExpressionNode]]:
        """Returns (set_id, apply) -- set_id is None for a time-denoting term."""
        shift_ctx = ctx.shift()  # type: ignore
        amount = shift_ctx.accept(self)  # type: ignore
        is_zero = expressions_equal(amount, literal(0))

        if shift_ctx.TIME() is not None:  # type: ignore
            if is_zero:
                return None, lambda expr: expr
            return None, lambda expr: expr.shift(amount)

        identifier: str = shift_ctx.IDENTIFIER().getText()  # type: ignore
        if self.identifiers.is_set(identifier):
            if is_zero:
                return identifier, lambda expr: expr.set_index(identifier)
            return identifier, lambda expr: expr.set_index(
                identifier, relative_shift=amount
            )

        # Legacy fallback: an ordinary parameter/variable identifier used as an
        # absolute time index, optionally offset -- preserves pre-existing
        # `X[some_param]` / `X[some_param+1]` semantics unchanged, for any
        # identifier that never resolves to a declared set.
        base = self._convert_identifier(identifier)
        eval_time = base if is_zero else base + amount
        return None, lambda expr: expr.eval(eval_time)

    # Visit a parse tree produced by ExprParser#keywordTerm.
    def visitKeywordTerm(
        self, ctx: ExprParser.KeywordTermContext
    ) -> Tuple[Optional[str], Callable[[ExpressionNode], ExpressionNode]]:
        """Returns (set_id, apply) -- set_id is None for a time-denoting term."""
        identifier: str = "t" if ctx.TIME() is not None else ctx.IDENTIFIER().getText()  # type: ignore
        comparator: str = ctx.COMPARISON().getText()  # type: ignore
        if comparator != "=":
            raise ValueError(
                f"'{comparator}' is not valid in the keyword index form "
                f"('{identifier}{comparator}...'); only '=' is."
            )
        position = ctx.expr().accept(self)  # type: ignore
        if identifier == "t":
            return None, lambda expr: expr.eval(position)
        if self.identifiers.is_set(identifier):
            return identifier, lambda expr: expr.set_index(
                identifier, position=position
            )
        raise ValueError(
            f"'{identifier}' is neither 't' nor a declared set; the keyword "
            f"index form ('{identifier}=...') is only valid for those."
        )

    # Visit a parse tree produced by ExprParser#positionTerm.
    def visitPositionTerm(
        self, ctx: ExprParser.PositionTermContext
    ) -> Tuple[Optional[str], Callable[[ExpressionNode], ExpressionNode]]:
        """Returns (set_id, apply) -- set_id is None for a time-denoting term."""
        eval_time = ctx.expr().accept(self)  # type: ignore
        return None, lambda expr: expr.eval(eval_time)

    def visitTimeSum(self, ctx: ExprParser.TimeSumContext) -> ExpressionNode:
        shifted_expr = ctx.expr().accept(self)  # type: ignore
        from_shift = ctx.from_.accept(self)  # type: ignore
        to_shift = ctx.to.accept(self)  # type: ignore
        return shifted_expr.time_sum(from_shift, to_shift)

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

        args: list[ExpressionNode] = (
            [expr.accept(self) for expr in arg_exprs]  # type: ignore
            if arg_list is not None
            else []
        )
        if function_name in _UNARY_FUNCTIONS:
            if len(args) != 1:
                raise ValueError(
                    f"Function {function_name} requires exactly 1 argument, got {len(args)}"
                )
            return _UNARY_FUNCTIONS[function_name](args[0])
        if function_name in _N_ARY_FUNCTIONS:
            return _N_ARY_FUNCTIONS[function_name](*args)
        raise ValueError(f"Encountered invalid function name {function_name}")

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
