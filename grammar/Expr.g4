/*
Copyright (c) 2024, RTE (https://www.rte-france.com)

See AUTHORS.txt

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

SPDX-License-Identifier: MPL-2.0

This file is part of the Antares project.
*/

grammar Expr;
portFieldExpr : IDENTIFIER '.' IDENTIFIER;
/* To match the whole input */
fullexpr: expr EOF;

expr
    : atom                                     # unsignedAtom
    | '-' expr                                 # negation
    | '(' expr ')'                             # expression
    | expr op=('/' | '*') expr                 # muldiv
    | expr op=('+' | '-') expr                 # addsub
    | expr COMPARISON expr                     # comparison
    | 'sum' '(' expr ')'                       # allTimeSum
    | 'sum_connections' '(' portFieldExpr ')'  # portFieldSum
    | 'sum' '(' from=sum_bound '..' to=sum_bound ',' expr ')'  # timeSum
    | IDENTIFIER '(' argList? ')'              # function
    | IDENTIFIER '[' shift ']'                 # timeShift
    | IDENTIFIER '[' time_expr ']'             # timeIndex
    | '(' expr ')' '[' shift ']'               # timeShiftExpr
    | '(' expr ')' '[' time_expr ']'           # timeIndexExpr
    ;

argList : expr (',' expr)* ;

// A time sum bound is either relative to the current time step ("t", "t + ...",
// "t - ...") or an absolute time index (an expression without "t").
sum_bound
    : shift
    | time_expr
    ;

atom
    : NUMBER                                   # number
    | IDENTIFIER                               # identifier
    ;

// A shift is "t" followed by additive terms: "t - d + 1" is t + (-d + 1).
shift: TIME (ops+=('+' | '-') terms+=time_term)*;

// Arithmetic of time shifts, time indices and sum bounds: numbers, parameters,
// parameters at a time index or time shift (d[0], d[t-1]), scalar functions,
// + - * / and parentheses. No variables, port fields, sums or comparisons.
// A unary minus only starts an expression: "-1", "t + (-1)", not "t + -1".
time_expr: neg='-'? time_term (ops+=('+' | '-') terms+=time_term)*;

time_term: time_factor (ops+=('*' | '/') factors+=time_factor)*;

time_factor
    : NUMBER                                         # timeNumber
    | IDENTIFIER                                     # timeIdentifier
    | IDENTIFIER '(' time_expr (',' time_expr)* ')'  # timeFunction
    | IDENTIFIER '[' shift ']'                       # timeFactorShift
    | IDENTIFIER '[' time_expr ']'                   # timeFactorIndex
    | '(' time_expr ')'                              # timeParenthesis
    ;


fragment DIGIT         : [0-9] ;
fragment CHAR          : [a-zA-Z_];
fragment CHAR_OR_DIGIT : (CHAR | DIGIT);

NUMBER        : DIGIT+ ('.' DIGIT+)?;
TIME          : 't';
IDENTIFIER    : CHAR CHAR_OR_DIGIT*;
COMPARISON    : ( '=' | '>=' | '<=' );

WS: (' ' | '\t' | '\r'| '\n') -> skip;
