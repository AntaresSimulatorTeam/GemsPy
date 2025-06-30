// Generated from /home/wsl/PlaneTerr/andromede-modeling-prototype/grammar/Expr.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast", "CheckReturnValue"})
public class ExprParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.13.1", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, NUMBER=14, TIME=15, MAX=16, IDENTIFIER=17, 
		COMPARISON=18, WS=19;
	public static final int
		RULE_portFieldExpr = 0, RULE_fullexpr = 1, RULE_expr = 2, RULE_atom = 3, 
		RULE_shift = 4, RULE_shift_expr = 5, RULE_right_expr = 6;
	private static String[] makeRuleNames() {
		return new String[] {
			"portFieldExpr", "fullexpr", "expr", "atom", "shift", "shift_expr", "right_expr"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'.'", "'-'", "'('", "')'", "'/'", "'*'", "'+'", "'sum'", "'sum_connections'", 
			"'..'", "','", "'['", "']'", null, "'t'", "'max'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, "NUMBER", "TIME", "MAX", "IDENTIFIER", "COMPARISON", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "Expr.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public ExprParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@SuppressWarnings("CheckReturnValue")
	public static class PortFieldExprContext extends ParserRuleContext {
		public List<TerminalNode> IDENTIFIER() { return getTokens(ExprParser.IDENTIFIER); }
		public TerminalNode IDENTIFIER(int i) {
			return getToken(ExprParser.IDENTIFIER, i);
		}
		public PortFieldExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_portFieldExpr; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterPortFieldExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitPortFieldExpr(this);
		}
	}

	public final PortFieldExprContext portFieldExpr() throws RecognitionException {
		PortFieldExprContext _localctx = new PortFieldExprContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_portFieldExpr);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(14);
			match(IDENTIFIER);
			setState(15);
			match(T__0);
			setState(16);
			match(IDENTIFIER);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class FullexprContext extends ParserRuleContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode EOF() { return getToken(ExprParser.EOF, 0); }
		public FullexprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_fullexpr; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterFullexpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitFullexpr(this);
		}
	}

	public final FullexprContext fullexpr() throws RecognitionException {
		FullexprContext _localctx = new FullexprContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_fullexpr);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(18);
			expr(0);
			setState(19);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ExprContext extends ParserRuleContext {
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	 
		public ExprContext() { }
		public void copyFrom(ExprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class PortFieldSumContext extends ExprContext {
		public PortFieldExprContext portFieldExpr() {
			return getRuleContext(PortFieldExprContext.class,0);
		}
		public PortFieldSumContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterPortFieldSum(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitPortFieldSum(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class NegationContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public NegationContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterNegation(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitNegation(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class UnsignedAtomContext extends ExprContext {
		public AtomContext atom() {
			return getRuleContext(AtomContext.class,0);
		}
		public UnsignedAtomContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterUnsignedAtom(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitUnsignedAtom(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ExpressionContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ExpressionContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitExpression(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ComparisonContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode COMPARISON() { return getToken(ExprParser.COMPARISON, 0); }
		public ComparisonContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterComparison(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitComparison(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AllTimeSumContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public AllTimeSumContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterAllTimeSum(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitAllTimeSum(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TimeIndexExprContext extends ExprContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TimeIndexExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterTimeIndexExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitTimeIndexExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class AddsubContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public AddsubContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterAddsub(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitAddsub(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TimeShiftExprContext extends ExprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public ShiftContext shift() {
			return getRuleContext(ShiftContext.class,0);
		}
		public TimeShiftExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterTimeShiftExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitTimeShiftExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class PortFieldContext extends ExprContext {
		public PortFieldExprContext portFieldExpr() {
			return getRuleContext(PortFieldExprContext.class,0);
		}
		public PortFieldContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterPortField(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitPortField(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MuldivContext extends ExprContext {
		public Token op;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MuldivContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterMuldiv(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitMuldiv(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TimeSumContext extends ExprContext {
		public ShiftContext from;
		public ShiftContext to;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<ShiftContext> shift() {
			return getRuleContexts(ShiftContext.class);
		}
		public ShiftContext shift(int i) {
			return getRuleContext(ShiftContext.class,i);
		}
		public TimeSumContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterTimeSum(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitTimeSum(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class MaxExprContext extends ExprContext {
		public TerminalNode MAX() { return getToken(ExprParser.MAX, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public MaxExprContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterMaxExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitMaxExpr(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TimeIndexContext extends ExprContext {
		public TerminalNode IDENTIFIER() { return getToken(ExprParser.IDENTIFIER, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TimeIndexContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterTimeIndex(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitTimeIndex(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class TimeShiftContext extends ExprContext {
		public TerminalNode IDENTIFIER() { return getToken(ExprParser.IDENTIFIER, 0); }
		public ShiftContext shift() {
			return getRuleContext(ShiftContext.class,0);
		}
		public TimeShiftContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterTimeShift(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitTimeShift(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class FunctionContext extends ExprContext {
		public TerminalNode IDENTIFIER() { return getToken(ExprParser.IDENTIFIER, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public FunctionContext(ExprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterFunction(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitFunction(this);
		}
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 4;
		enterRecursionRule(_localctx, 4, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(90);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,1,_ctx) ) {
			case 1:
				{
				_localctx = new UnsignedAtomContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(22);
				atom();
				}
				break;
			case 2:
				{
				_localctx = new PortFieldContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(23);
				portFieldExpr();
				}
				break;
			case 3:
				{
				_localctx = new NegationContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(24);
				match(T__1);
				setState(25);
				expr(14);
				}
				break;
			case 4:
				{
				_localctx = new ExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(26);
				match(T__2);
				setState(27);
				expr(0);
				setState(28);
				match(T__3);
				}
				break;
			case 5:
				{
				_localctx = new AllTimeSumContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(30);
				match(T__7);
				setState(31);
				match(T__2);
				setState(32);
				expr(0);
				setState(33);
				match(T__3);
				}
				break;
			case 6:
				{
				_localctx = new PortFieldSumContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(35);
				match(T__8);
				setState(36);
				match(T__2);
				setState(37);
				portFieldExpr();
				setState(38);
				match(T__3);
				}
				break;
			case 7:
				{
				_localctx = new TimeSumContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(40);
				match(T__7);
				setState(41);
				match(T__2);
				setState(42);
				((TimeSumContext)_localctx).from = shift();
				setState(43);
				match(T__9);
				setState(44);
				((TimeSumContext)_localctx).to = shift();
				setState(45);
				match(T__10);
				setState(46);
				expr(0);
				setState(47);
				match(T__3);
				}
				break;
			case 8:
				{
				_localctx = new FunctionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(49);
				match(IDENTIFIER);
				setState(50);
				match(T__2);
				setState(51);
				expr(0);
				setState(52);
				match(T__3);
				}
				break;
			case 9:
				{
				_localctx = new TimeShiftContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(54);
				match(IDENTIFIER);
				setState(55);
				match(T__11);
				setState(56);
				shift();
				setState(57);
				match(T__12);
				}
				break;
			case 10:
				{
				_localctx = new TimeIndexContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(59);
				match(IDENTIFIER);
				setState(60);
				match(T__11);
				setState(61);
				expr(0);
				setState(62);
				match(T__12);
				}
				break;
			case 11:
				{
				_localctx = new TimeShiftExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(64);
				match(T__2);
				setState(65);
				expr(0);
				setState(66);
				match(T__3);
				setState(67);
				match(T__11);
				setState(68);
				shift();
				setState(69);
				match(T__12);
				}
				break;
			case 12:
				{
				_localctx = new TimeIndexExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(71);
				match(T__2);
				setState(72);
				expr(0);
				setState(73);
				match(T__3);
				setState(74);
				match(T__11);
				setState(75);
				expr(0);
				setState(76);
				match(T__12);
				}
				break;
			case 13:
				{
				_localctx = new MaxExprContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(78);
				match(MAX);
				setState(79);
				match(T__2);
				setState(80);
				expr(0);
				setState(85);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==T__10) {
					{
					{
					setState(81);
					match(T__10);
					setState(82);
					expr(0);
					}
					}
					setState(87);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				setState(88);
				match(T__3);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(103);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,3,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(101);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
					case 1:
						{
						_localctx = new MuldivContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(92);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(93);
						((MuldivContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__4 || _la==T__5) ) {
							((MuldivContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(94);
						expr(13);
						}
						break;
					case 2:
						{
						_localctx = new AddsubContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(95);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(96);
						((AddsubContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__1 || _la==T__6) ) {
							((AddsubContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(97);
						expr(12);
						}
						break;
					case 3:
						{
						_localctx = new ComparisonContext(new ExprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(98);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(99);
						match(COMPARISON);
						setState(100);
						expr(11);
						}
						break;
					}
					} 
				}
				setState(105);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,3,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class AtomContext extends ParserRuleContext {
		public AtomContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_atom; }
	 
		public AtomContext() { }
		public void copyFrom(AtomContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class NumberContext extends AtomContext {
		public TerminalNode NUMBER() { return getToken(ExprParser.NUMBER, 0); }
		public NumberContext(AtomContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterNumber(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitNumber(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class IdentifierContext extends AtomContext {
		public TerminalNode IDENTIFIER() { return getToken(ExprParser.IDENTIFIER, 0); }
		public IdentifierContext(AtomContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterIdentifier(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitIdentifier(this);
		}
	}

	public final AtomContext atom() throws RecognitionException {
		AtomContext _localctx = new AtomContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_atom);
		try {
			setState(108);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case NUMBER:
				_localctx = new NumberContext(_localctx);
				enterOuterAlt(_localctx, 1);
				{
				setState(106);
				match(NUMBER);
				}
				break;
			case IDENTIFIER:
				_localctx = new IdentifierContext(_localctx);
				enterOuterAlt(_localctx, 2);
				{
				setState(107);
				match(IDENTIFIER);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class ShiftContext extends ParserRuleContext {
		public TerminalNode TIME() { return getToken(ExprParser.TIME, 0); }
		public Shift_exprContext shift_expr() {
			return getRuleContext(Shift_exprContext.class,0);
		}
		public ShiftContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_shift; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterShift(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitShift(this);
		}
	}

	public final ShiftContext shift() throws RecognitionException {
		ShiftContext _localctx = new ShiftContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_shift);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(110);
			match(TIME);
			setState(112);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==T__1 || _la==T__6) {
				{
				setState(111);
				shift_expr(0);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Shift_exprContext extends ParserRuleContext {
		public Shift_exprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_shift_expr; }
	 
		public Shift_exprContext() { }
		public void copyFrom(Shift_exprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class SignedAtomContext extends Shift_exprContext {
		public Token op;
		public AtomContext atom() {
			return getRuleContext(AtomContext.class,0);
		}
		public SignedAtomContext(Shift_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterSignedAtom(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitSignedAtom(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class SignedExpressionContext extends Shift_exprContext {
		public Token op;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public SignedExpressionContext(Shift_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterSignedExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitSignedExpression(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ShiftMuldivContext extends Shift_exprContext {
		public Token op;
		public Shift_exprContext shift_expr() {
			return getRuleContext(Shift_exprContext.class,0);
		}
		public Right_exprContext right_expr() {
			return getRuleContext(Right_exprContext.class,0);
		}
		public ShiftMuldivContext(Shift_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterShiftMuldiv(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitShiftMuldiv(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class ShiftAddsubContext extends Shift_exprContext {
		public Token op;
		public Shift_exprContext shift_expr() {
			return getRuleContext(Shift_exprContext.class,0);
		}
		public Right_exprContext right_expr() {
			return getRuleContext(Right_exprContext.class,0);
		}
		public ShiftAddsubContext(Shift_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterShiftAddsub(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitShiftAddsub(this);
		}
	}

	public final Shift_exprContext shift_expr() throws RecognitionException {
		return shift_expr(0);
	}

	private Shift_exprContext shift_expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		Shift_exprContext _localctx = new Shift_exprContext(_ctx, _parentState);
		Shift_exprContext _prevctx = _localctx;
		int _startState = 10;
		enterRecursionRule(_localctx, 10, RULE_shift_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(122);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
			case 1:
				{
				_localctx = new SignedAtomContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(115);
				((SignedAtomContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !(_la==T__1 || _la==T__6) ) {
					((SignedAtomContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(116);
				atom();
				}
				break;
			case 2:
				{
				_localctx = new SignedExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(117);
				((SignedExpressionContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !(_la==T__1 || _la==T__6) ) {
					((SignedExpressionContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(118);
				match(T__2);
				setState(119);
				expr(0);
				setState(120);
				match(T__3);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(132);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,8,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(130);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
					case 1:
						{
						_localctx = new ShiftMuldivContext(new Shift_exprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_shift_expr);
						setState(124);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(125);
						((ShiftMuldivContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__4 || _la==T__5) ) {
							((ShiftMuldivContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(126);
						right_expr(0);
						}
						break;
					case 2:
						{
						_localctx = new ShiftAddsubContext(new Shift_exprContext(_parentctx, _parentState));
						pushNewRecursionContext(_localctx, _startState, RULE_shift_expr);
						setState(127);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(128);
						((ShiftAddsubContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==T__1 || _la==T__6) ) {
							((ShiftAddsubContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(129);
						right_expr(0);
						}
						break;
					}
					} 
				}
				setState(134);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,8,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	@SuppressWarnings("CheckReturnValue")
	public static class Right_exprContext extends ParserRuleContext {
		public Right_exprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_right_expr; }
	 
		public Right_exprContext() { }
		public void copyFrom(Right_exprContext ctx) {
			super.copyFrom(ctx);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RightExpressionContext extends Right_exprContext {
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public RightExpressionContext(Right_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterRightExpression(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitRightExpression(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RightMuldivContext extends Right_exprContext {
		public Token op;
		public List<Right_exprContext> right_expr() {
			return getRuleContexts(Right_exprContext.class);
		}
		public Right_exprContext right_expr(int i) {
			return getRuleContext(Right_exprContext.class,i);
		}
		public RightMuldivContext(Right_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterRightMuldiv(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitRightMuldiv(this);
		}
	}
	@SuppressWarnings("CheckReturnValue")
	public static class RightAtomContext extends Right_exprContext {
		public AtomContext atom() {
			return getRuleContext(AtomContext.class,0);
		}
		public RightAtomContext(Right_exprContext ctx) { copyFrom(ctx); }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).enterRightAtom(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof ExprListener ) ((ExprListener)listener).exitRightAtom(this);
		}
	}

	public final Right_exprContext right_expr() throws RecognitionException {
		return right_expr(0);
	}

	private Right_exprContext right_expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		Right_exprContext _localctx = new Right_exprContext(_ctx, _parentState);
		Right_exprContext _prevctx = _localctx;
		int _startState = 12;
		enterRecursionRule(_localctx, 12, RULE_right_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(141);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__2:
				{
				_localctx = new RightExpressionContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;

				setState(136);
				match(T__2);
				setState(137);
				expr(0);
				setState(138);
				match(T__3);
				}
				break;
			case NUMBER:
			case IDENTIFIER:
				{
				_localctx = new RightAtomContext(_localctx);
				_ctx = _localctx;
				_prevctx = _localctx;
				setState(140);
				atom();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(148);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new RightMuldivContext(new Right_exprContext(_parentctx, _parentState));
					pushNewRecursionContext(_localctx, _startState, RULE_right_expr);
					setState(143);
					if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
					setState(144);
					((RightMuldivContext)_localctx).op = _input.LT(1);
					_la = _input.LA(1);
					if ( !(_la==T__4 || _la==T__5) ) {
						((RightMuldivContext)_localctx).op = (Token)_errHandler.recoverInline(this);
					}
					else {
						if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
						_errHandler.reportMatch(this);
						consume();
					}
					setState(145);
					right_expr(4);
					}
					} 
				}
				setState(150);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,10,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 2:
			return expr_sempred((ExprContext)_localctx, predIndex);
		case 5:
			return shift_expr_sempred((Shift_exprContext)_localctx, predIndex);
		case 6:
			return right_expr_sempred((Right_exprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 12);
		case 1:
			return precpred(_ctx, 11);
		case 2:
			return precpred(_ctx, 10);
		}
		return true;
	}
	private boolean shift_expr_sempred(Shift_exprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 3:
			return precpred(_ctx, 4);
		case 4:
			return precpred(_ctx, 3);
		}
		return true;
	}
	private boolean right_expr_sempred(Right_exprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 5:
			return precpred(_ctx, 3);
		}
		return true;
	}

	public static final String _serializedATN =
		"\u0004\u0001\u0013\u0098\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001"+
		"\u0002\u0002\u0007\u0002\u0002\u0003\u0007\u0003\u0002\u0004\u0007\u0004"+
		"\u0002\u0005\u0007\u0005\u0002\u0006\u0007\u0006\u0001\u0000\u0001\u0000"+
		"\u0001\u0000\u0001\u0000\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0005\u0002T\b\u0002\n\u0002\f\u0002W\t\u0002\u0001\u0002"+
		"\u0001\u0002\u0003\u0002[\b\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002\u0001\u0002"+
		"\u0005\u0002f\b\u0002\n\u0002\f\u0002i\t\u0002\u0001\u0003\u0001\u0003"+
		"\u0003\u0003m\b\u0003\u0001\u0004\u0001\u0004\u0003\u0004q\b\u0004\u0001"+
		"\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0005\u0001\u0005\u0003\u0005{\b\u0005\u0001\u0005\u0001\u0005\u0001"+
		"\u0005\u0001\u0005\u0001\u0005\u0001\u0005\u0005\u0005\u0083\b\u0005\n"+
		"\u0005\f\u0005\u0086\t\u0005\u0001\u0006\u0001\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0001\u0006\u0003\u0006\u008e\b\u0006\u0001\u0006\u0001"+
		"\u0006\u0001\u0006\u0005\u0006\u0093\b\u0006\n\u0006\f\u0006\u0096\t\u0006"+
		"\u0001\u0006\u0000\u0003\u0004\n\f\u0007\u0000\u0002\u0004\u0006\b\n\f"+
		"\u0000\u0002\u0001\u0000\u0005\u0006\u0002\u0000\u0002\u0002\u0007\u0007"+
		"\u00a7\u0000\u000e\u0001\u0000\u0000\u0000\u0002\u0012\u0001\u0000\u0000"+
		"\u0000\u0004Z\u0001\u0000\u0000\u0000\u0006l\u0001\u0000\u0000\u0000\b"+
		"n\u0001\u0000\u0000\u0000\nz\u0001\u0000\u0000\u0000\f\u008d\u0001\u0000"+
		"\u0000\u0000\u000e\u000f\u0005\u0011\u0000\u0000\u000f\u0010\u0005\u0001"+
		"\u0000\u0000\u0010\u0011\u0005\u0011\u0000\u0000\u0011\u0001\u0001\u0000"+
		"\u0000\u0000\u0012\u0013\u0003\u0004\u0002\u0000\u0013\u0014\u0005\u0000"+
		"\u0000\u0001\u0014\u0003\u0001\u0000\u0000\u0000\u0015\u0016\u0006\u0002"+
		"\uffff\uffff\u0000\u0016[\u0003\u0006\u0003\u0000\u0017[\u0003\u0000\u0000"+
		"\u0000\u0018\u0019\u0005\u0002\u0000\u0000\u0019[\u0003\u0004\u0002\u000e"+
		"\u001a\u001b\u0005\u0003\u0000\u0000\u001b\u001c\u0003\u0004\u0002\u0000"+
		"\u001c\u001d\u0005\u0004\u0000\u0000\u001d[\u0001\u0000\u0000\u0000\u001e"+
		"\u001f\u0005\b\u0000\u0000\u001f \u0005\u0003\u0000\u0000 !\u0003\u0004"+
		"\u0002\u0000!\"\u0005\u0004\u0000\u0000\"[\u0001\u0000\u0000\u0000#$\u0005"+
		"\t\u0000\u0000$%\u0005\u0003\u0000\u0000%&\u0003\u0000\u0000\u0000&\'"+
		"\u0005\u0004\u0000\u0000\'[\u0001\u0000\u0000\u0000()\u0005\b\u0000\u0000"+
		")*\u0005\u0003\u0000\u0000*+\u0003\b\u0004\u0000+,\u0005\n\u0000\u0000"+
		",-\u0003\b\u0004\u0000-.\u0005\u000b\u0000\u0000./\u0003\u0004\u0002\u0000"+
		"/0\u0005\u0004\u0000\u00000[\u0001\u0000\u0000\u000012\u0005\u0011\u0000"+
		"\u000023\u0005\u0003\u0000\u000034\u0003\u0004\u0002\u000045\u0005\u0004"+
		"\u0000\u00005[\u0001\u0000\u0000\u000067\u0005\u0011\u0000\u000078\u0005"+
		"\f\u0000\u000089\u0003\b\u0004\u00009:\u0005\r\u0000\u0000:[\u0001\u0000"+
		"\u0000\u0000;<\u0005\u0011\u0000\u0000<=\u0005\f\u0000\u0000=>\u0003\u0004"+
		"\u0002\u0000>?\u0005\r\u0000\u0000?[\u0001\u0000\u0000\u0000@A\u0005\u0003"+
		"\u0000\u0000AB\u0003\u0004\u0002\u0000BC\u0005\u0004\u0000\u0000CD\u0005"+
		"\f\u0000\u0000DE\u0003\b\u0004\u0000EF\u0005\r\u0000\u0000F[\u0001\u0000"+
		"\u0000\u0000GH\u0005\u0003\u0000\u0000HI\u0003\u0004\u0002\u0000IJ\u0005"+
		"\u0004\u0000\u0000JK\u0005\f\u0000\u0000KL\u0003\u0004\u0002\u0000LM\u0005"+
		"\r\u0000\u0000M[\u0001\u0000\u0000\u0000NO\u0005\u0010\u0000\u0000OP\u0005"+
		"\u0003\u0000\u0000PU\u0003\u0004\u0002\u0000QR\u0005\u000b\u0000\u0000"+
		"RT\u0003\u0004\u0002\u0000SQ\u0001\u0000\u0000\u0000TW\u0001\u0000\u0000"+
		"\u0000US\u0001\u0000\u0000\u0000UV\u0001\u0000\u0000\u0000VX\u0001\u0000"+
		"\u0000\u0000WU\u0001\u0000\u0000\u0000XY\u0005\u0004\u0000\u0000Y[\u0001"+
		"\u0000\u0000\u0000Z\u0015\u0001\u0000\u0000\u0000Z\u0017\u0001\u0000\u0000"+
		"\u0000Z\u0018\u0001\u0000\u0000\u0000Z\u001a\u0001\u0000\u0000\u0000Z"+
		"\u001e\u0001\u0000\u0000\u0000Z#\u0001\u0000\u0000\u0000Z(\u0001\u0000"+
		"\u0000\u0000Z1\u0001\u0000\u0000\u0000Z6\u0001\u0000\u0000\u0000Z;\u0001"+
		"\u0000\u0000\u0000Z@\u0001\u0000\u0000\u0000ZG\u0001\u0000\u0000\u0000"+
		"ZN\u0001\u0000\u0000\u0000[g\u0001\u0000\u0000\u0000\\]\n\f\u0000\u0000"+
		"]^\u0007\u0000\u0000\u0000^f\u0003\u0004\u0002\r_`\n\u000b\u0000\u0000"+
		"`a\u0007\u0001\u0000\u0000af\u0003\u0004\u0002\fbc\n\n\u0000\u0000cd\u0005"+
		"\u0012\u0000\u0000df\u0003\u0004\u0002\u000be\\\u0001\u0000\u0000\u0000"+
		"e_\u0001\u0000\u0000\u0000eb\u0001\u0000\u0000\u0000fi\u0001\u0000\u0000"+
		"\u0000ge\u0001\u0000\u0000\u0000gh\u0001\u0000\u0000\u0000h\u0005\u0001"+
		"\u0000\u0000\u0000ig\u0001\u0000\u0000\u0000jm\u0005\u000e\u0000\u0000"+
		"km\u0005\u0011\u0000\u0000lj\u0001\u0000\u0000\u0000lk\u0001\u0000\u0000"+
		"\u0000m\u0007\u0001\u0000\u0000\u0000np\u0005\u000f\u0000\u0000oq\u0003"+
		"\n\u0005\u0000po\u0001\u0000\u0000\u0000pq\u0001\u0000\u0000\u0000q\t"+
		"\u0001\u0000\u0000\u0000rs\u0006\u0005\uffff\uffff\u0000st\u0007\u0001"+
		"\u0000\u0000t{\u0003\u0006\u0003\u0000uv\u0007\u0001\u0000\u0000vw\u0005"+
		"\u0003\u0000\u0000wx\u0003\u0004\u0002\u0000xy\u0005\u0004\u0000\u0000"+
		"y{\u0001\u0000\u0000\u0000zr\u0001\u0000\u0000\u0000zu\u0001\u0000\u0000"+
		"\u0000{\u0084\u0001\u0000\u0000\u0000|}\n\u0004\u0000\u0000}~\u0007\u0000"+
		"\u0000\u0000~\u0083\u0003\f\u0006\u0000\u007f\u0080\n\u0003\u0000\u0000"+
		"\u0080\u0081\u0007\u0001\u0000\u0000\u0081\u0083\u0003\f\u0006\u0000\u0082"+
		"|\u0001\u0000\u0000\u0000\u0082\u007f\u0001\u0000\u0000\u0000\u0083\u0086"+
		"\u0001\u0000\u0000\u0000\u0084\u0082\u0001\u0000\u0000\u0000\u0084\u0085"+
		"\u0001\u0000\u0000\u0000\u0085\u000b\u0001\u0000\u0000\u0000\u0086\u0084"+
		"\u0001\u0000\u0000\u0000\u0087\u0088\u0006\u0006\uffff\uffff\u0000\u0088"+
		"\u0089\u0005\u0003\u0000\u0000\u0089\u008a\u0003\u0004\u0002\u0000\u008a"+
		"\u008b\u0005\u0004\u0000\u0000\u008b\u008e\u0001\u0000\u0000\u0000\u008c"+
		"\u008e\u0003\u0006\u0003\u0000\u008d\u0087\u0001\u0000\u0000\u0000\u008d"+
		"\u008c\u0001\u0000\u0000\u0000\u008e\u0094\u0001\u0000\u0000\u0000\u008f"+
		"\u0090\n\u0003\u0000\u0000\u0090\u0091\u0007\u0000\u0000\u0000\u0091\u0093"+
		"\u0003\f\u0006\u0004\u0092\u008f\u0001\u0000\u0000\u0000\u0093\u0096\u0001"+
		"\u0000\u0000\u0000\u0094\u0092\u0001\u0000\u0000\u0000\u0094\u0095\u0001"+
		"\u0000\u0000\u0000\u0095\r\u0001\u0000\u0000\u0000\u0096\u0094\u0001\u0000"+
		"\u0000\u0000\u000bUZeglpz\u0082\u0084\u008d\u0094";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}