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
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.Token import Token

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
    sets: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        clashes = set(self.sets) & (set(self.variables) | set(self.parameters))
        if clashes:
            raise ValueError(
                f"Set ids clash with variable/parameter ids: {sorted(clashes)}"
            )
        if "t" in self.sets:
            raise ValueError(
                "'t' is reserved for the time dimension and cannot be a set id."
            )

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
        if self.identifiers.is_set(identifier):
            raise ValueError(
                f"'{identifier}' is a set, not a valid variable or parameter name; "
                f"a set can only be used as X[{identifier}], X[{identifier}+k], "
                f"X[{identifier}=k] or sum_over({identifier}, ...)."
            )
        message = f"{identifier} is not a valid variable or parameter name."
        if self.identifiers.sets:
            message += f" (declared sets: {sorted(self.identifiers.sets)})"
        raise ValueError(message)

    # Visit a parse tree produced by ExprParser#comparison.
    def visitComparison(self, ctx: ExprParser.ComparisonContext) -> ExpressionNode:
        op = ctx.op.text  # type: ignore
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
        time_term_text = ""
        set_appliers: Dict[str, Callable[[ExpressionNode], ExpressionNode]] = {}

        for term_ctx in index_list_ctx.indexTerm():  # type: ignore
            set_id, apply = term_ctx.accept(self)  # type: ignore
            if set_id is None:
                if time_applier is not None:
                    raise ValueError(
                        "An index list cannot contain more than one term denoting "
                        f"the time dimension: got '{time_term_text}' and "
                        f"'{term_ctx.getText()}' in '[{index_list_ctx.getText()}]'."
                    )
                time_applier = apply
                time_term_text = term_ctx.getText()
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

    # Visit a parse tree produced by ExprParser#timeShiftTerm.
    def visitTimeShiftTerm(
        self, ctx: ExprParser.TimeShiftTermContext
    ) -> Tuple[Optional[str], Callable[[ExpressionNode], ExpressionNode]]:
        """Returns (set_id, apply) -- set_id is None for a time-denoting term."""
        amount = ctx.shift().accept(self)  # type: ignore
        return None, lambda expr: _shift(expr, amount)

    # Visit a parse tree produced by ExprParser#keywordTerm.
    def visitKeywordTerm(
        self, ctx: ExprParser.KeywordTermContext
    ) -> Tuple[Optional[str], Callable[[ExpressionNode], ExpressionNode]]:
        """Returns (set_id, apply) -- set_id is None for a time-denoting term."""
        identifier: str = "t" if ctx.TIME() is not None else ctx.IDENTIFIER().getText()  # type: ignore
        position = ctx.time_expr().accept(self)  # type: ignore
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
        """Returns (set_id, apply) -- set_id is None for a time-denoting term.

        `set`, `set + ...` or `set - ...` (with `set` a declared set) is a set
        shift; any other expression is an absolute time index.
        """
        time_expr = ctx.time_expr()  # type: ignore
        set_id = self._leading_set_id(time_expr)
        if set_id is None:
            eval_time = time_expr.accept(self)
            return None, lambda expr: expr.eval(eval_time)

        terms = [term.accept(self) for term in time_expr.time_term()[1:]]
        ops = time_expr.ops
        amount: ExpressionNode = literal(0)
        if terms:
            first = -terms[0] if ops[0].text == "-" else terms[0]
            amount = _fold(first, ops[1:], terms[1:])
        if expressions_equal(amount, literal(0)):
            return set_id, lambda expr: expr.set_index(set_id)
        return set_id, lambda expr: expr.set_index(set_id, relative_shift=amount)

    def _leading_set_id(self, ctx: ExprParser.Time_exprContext) -> Optional[str]:
        """The set id if ctx starts with a bare declared set id, else None."""
        if ctx.neg is not None:  # type: ignore
            return None
        factors = ctx.time_term(0).time_factor()  # type: ignore
        if len(factors) != 1 or not isinstance(
            factors[0], ExprParser.TimeIdentifierContext
        ):
            return None
        identifier: str = factors[0].IDENTIFIER().getText()  # type: ignore
        return identifier if self.identifiers.is_set(identifier) else None

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
        ops = ctx.ops  # type: ignore
        first = -terms[0] if ops[0].text == "-" else terms[0]
        return _fold(first, ops[1:], terms[1:])

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
    first: ExpressionNode, ops: Sequence[Token], operands: Sequence[ExpressionNode]
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
