// Generated from /home/wsl/PlaneTerr/andromede-modeling-prototype/grammar/Expr.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link ExprParser}.
 */
public interface ExprListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link ExprParser#portFieldExpr}.
	 * @param ctx the parse tree
	 */
	void enterPortFieldExpr(ExprParser.PortFieldExprContext ctx);
	/**
	 * Exit a parse tree produced by {@link ExprParser#portFieldExpr}.
	 * @param ctx the parse tree
	 */
	void exitPortFieldExpr(ExprParser.PortFieldExprContext ctx);
	/**
	 * Enter a parse tree produced by {@link ExprParser#fullexpr}.
	 * @param ctx the parse tree
	 */
	void enterFullexpr(ExprParser.FullexprContext ctx);
	/**
	 * Exit a parse tree produced by {@link ExprParser#fullexpr}.
	 * @param ctx the parse tree
	 */
	void exitFullexpr(ExprParser.FullexprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code portFieldSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterPortFieldSum(ExprParser.PortFieldSumContext ctx);
	/**
	 * Exit a parse tree produced by the {@code portFieldSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitPortFieldSum(ExprParser.PortFieldSumContext ctx);
	/**
	 * Enter a parse tree produced by the {@code negation}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterNegation(ExprParser.NegationContext ctx);
	/**
	 * Exit a parse tree produced by the {@code negation}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitNegation(ExprParser.NegationContext ctx);
	/**
	 * Enter a parse tree produced by the {@code unsignedAtom}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterUnsignedAtom(ExprParser.UnsignedAtomContext ctx);
	/**
	 * Exit a parse tree produced by the {@code unsignedAtom}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitUnsignedAtom(ExprParser.UnsignedAtomContext ctx);
	/**
	 * Enter a parse tree produced by the {@code expression}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterExpression(ExprParser.ExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code expression}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitExpression(ExprParser.ExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code comparison}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterComparison(ExprParser.ComparisonContext ctx);
	/**
	 * Exit a parse tree produced by the {@code comparison}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitComparison(ExprParser.ComparisonContext ctx);
	/**
	 * Enter a parse tree produced by the {@code allTimeSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterAllTimeSum(ExprParser.AllTimeSumContext ctx);
	/**
	 * Exit a parse tree produced by the {@code allTimeSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitAllTimeSum(ExprParser.AllTimeSumContext ctx);
	/**
	 * Enter a parse tree produced by the {@code timeIndexExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTimeIndexExpr(ExprParser.TimeIndexExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code timeIndexExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTimeIndexExpr(ExprParser.TimeIndexExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code addsub}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterAddsub(ExprParser.AddsubContext ctx);
	/**
	 * Exit a parse tree produced by the {@code addsub}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitAddsub(ExprParser.AddsubContext ctx);
	/**
	 * Enter a parse tree produced by the {@code timeShiftExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTimeShiftExpr(ExprParser.TimeShiftExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code timeShiftExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTimeShiftExpr(ExprParser.TimeShiftExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code portField}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterPortField(ExprParser.PortFieldContext ctx);
	/**
	 * Exit a parse tree produced by the {@code portField}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitPortField(ExprParser.PortFieldContext ctx);
	/**
	 * Enter a parse tree produced by the {@code muldiv}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMuldiv(ExprParser.MuldivContext ctx);
	/**
	 * Exit a parse tree produced by the {@code muldiv}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMuldiv(ExprParser.MuldivContext ctx);
	/**
	 * Enter a parse tree produced by the {@code timeSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTimeSum(ExprParser.TimeSumContext ctx);
	/**
	 * Exit a parse tree produced by the {@code timeSum}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTimeSum(ExprParser.TimeSumContext ctx);
	/**
	 * Enter a parse tree produced by the {@code maxExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterMaxExpr(ExprParser.MaxExprContext ctx);
	/**
	 * Exit a parse tree produced by the {@code maxExpr}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitMaxExpr(ExprParser.MaxExprContext ctx);
	/**
	 * Enter a parse tree produced by the {@code timeIndex}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTimeIndex(ExprParser.TimeIndexContext ctx);
	/**
	 * Exit a parse tree produced by the {@code timeIndex}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTimeIndex(ExprParser.TimeIndexContext ctx);
	/**
	 * Enter a parse tree produced by the {@code timeShift}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterTimeShift(ExprParser.TimeShiftContext ctx);
	/**
	 * Exit a parse tree produced by the {@code timeShift}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitTimeShift(ExprParser.TimeShiftContext ctx);
	/**
	 * Enter a parse tree produced by the {@code function}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterFunction(ExprParser.FunctionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code function}
	 * labeled alternative in {@link ExprParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitFunction(ExprParser.FunctionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code number}
	 * labeled alternative in {@link ExprParser#atom}.
	 * @param ctx the parse tree
	 */
	void enterNumber(ExprParser.NumberContext ctx);
	/**
	 * Exit a parse tree produced by the {@code number}
	 * labeled alternative in {@link ExprParser#atom}.
	 * @param ctx the parse tree
	 */
	void exitNumber(ExprParser.NumberContext ctx);
	/**
	 * Enter a parse tree produced by the {@code identifier}
	 * labeled alternative in {@link ExprParser#atom}.
	 * @param ctx the parse tree
	 */
	void enterIdentifier(ExprParser.IdentifierContext ctx);
	/**
	 * Exit a parse tree produced by the {@code identifier}
	 * labeled alternative in {@link ExprParser#atom}.
	 * @param ctx the parse tree
	 */
	void exitIdentifier(ExprParser.IdentifierContext ctx);
	/**
	 * Enter a parse tree produced by {@link ExprParser#shift}.
	 * @param ctx the parse tree
	 */
	void enterShift(ExprParser.ShiftContext ctx);
	/**
	 * Exit a parse tree produced by {@link ExprParser#shift}.
	 * @param ctx the parse tree
	 */
	void exitShift(ExprParser.ShiftContext ctx);
	/**
	 * Enter a parse tree produced by the {@code signedAtom}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void enterSignedAtom(ExprParser.SignedAtomContext ctx);
	/**
	 * Exit a parse tree produced by the {@code signedAtom}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void exitSignedAtom(ExprParser.SignedAtomContext ctx);
	/**
	 * Enter a parse tree produced by the {@code signedExpression}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void enterSignedExpression(ExprParser.SignedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code signedExpression}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void exitSignedExpression(ExprParser.SignedExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code shiftMuldiv}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void enterShiftMuldiv(ExprParser.ShiftMuldivContext ctx);
	/**
	 * Exit a parse tree produced by the {@code shiftMuldiv}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void exitShiftMuldiv(ExprParser.ShiftMuldivContext ctx);
	/**
	 * Enter a parse tree produced by the {@code shiftAddsub}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void enterShiftAddsub(ExprParser.ShiftAddsubContext ctx);
	/**
	 * Exit a parse tree produced by the {@code shiftAddsub}
	 * labeled alternative in {@link ExprParser#shift_expr}.
	 * @param ctx the parse tree
	 */
	void exitShiftAddsub(ExprParser.ShiftAddsubContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rightExpression}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void enterRightExpression(ExprParser.RightExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rightExpression}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void exitRightExpression(ExprParser.RightExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rightMuldiv}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void enterRightMuldiv(ExprParser.RightMuldivContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rightMuldiv}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void exitRightMuldiv(ExprParser.RightMuldivContext ctx);
	/**
	 * Enter a parse tree produced by the {@code rightAtom}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void enterRightAtom(ExprParser.RightAtomContext ctx);
	/**
	 * Exit a parse tree produced by the {@code rightAtom}
	 * labeled alternative in {@link ExprParser#right_expr}.
	 * @param ctx the parse tree
	 */
	void exitRightAtom(ExprParser.RightAtomContext ctx);
}