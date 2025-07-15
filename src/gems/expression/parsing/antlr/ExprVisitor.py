# Generated from Expr.g4 by ANTLR 4.13.2
from antlr4 import *

if "." in __name__:
    from .ExprParser import ExprParser
else:
    from ExprParser import ExprParser

# This class defines a complete generic visitor for a parse tree produced by ExprParser.


class ExprVisitor(ParseTreeVisitor):
    # Visit a parse tree produced by ExprParser#portFieldExpr.
    def visitPortFieldExpr(self, ctx: ExprParser.PortFieldExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#fullexpr.
    def visitFullexpr(self, ctx: ExprParser.FullexprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#portFieldSum.
    def visitPortFieldSum(self, ctx: ExprParser.PortFieldSumContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#negation.
    def visitNegation(self, ctx: ExprParser.NegationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#unsignedAtom.
    def visitUnsignedAtom(self, ctx: ExprParser.UnsignedAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#expression.
    def visitExpression(self, ctx: ExprParser.ExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#comparison.
    def visitComparison(self, ctx: ExprParser.ComparisonContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#allTimeSum.
    def visitAllTimeSum(self, ctx: ExprParser.AllTimeSumContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#timeIndexExpr.
    def visitTimeIndexExpr(self, ctx: ExprParser.TimeIndexExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#addsub.
    def visitAddsub(self, ctx: ExprParser.AddsubContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#timeShiftExpr.
    def visitTimeShiftExpr(self, ctx: ExprParser.TimeShiftExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#portField.
    def visitPortField(self, ctx: ExprParser.PortFieldContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#muldiv.
    def visitMuldiv(self, ctx: ExprParser.MuldivContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#timeSum.
    def visitTimeSum(self, ctx: ExprParser.TimeSumContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#timeIndex.
    def visitTimeIndex(self, ctx: ExprParser.TimeIndexContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#timeShift.
    def visitTimeShift(self, ctx: ExprParser.TimeShiftContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#function.
    def visitFunction(self, ctx: ExprParser.FunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#number.
    def visitNumber(self, ctx: ExprParser.NumberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#identifier.
    def visitIdentifier(self, ctx: ExprParser.IdentifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#shift.
    def visitShift(self, ctx: ExprParser.ShiftContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#signedAtom.
    def visitSignedAtom(self, ctx: ExprParser.SignedAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#signedExpression.
    def visitSignedExpression(self, ctx: ExprParser.SignedExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#shiftMuldiv.
    def visitShiftMuldiv(self, ctx: ExprParser.ShiftMuldivContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#shiftAddsub.
    def visitShiftAddsub(self, ctx: ExprParser.ShiftAddsubContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#rightExpression.
    def visitRightExpression(self, ctx: ExprParser.RightExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#rightMuldiv.
    def visitRightMuldiv(self, ctx: ExprParser.RightMuldivContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by ExprParser#rightAtom.
    def visitRightAtom(self, ctx: ExprParser.RightAtomContext):
        return self.visitChildren(ctx)


del ExprParser
