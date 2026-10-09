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
    | portFieldExpr                            # portField
    | <assoc=right> expr '^' expr              # power
    | '-' expr                                 # negation
    | '(' expr ')'                             # expression
    | expr op=('/' | '*') expr                 # muldiv
    | expr op=('+' | '-') expr                 # addsub
    | expr COMPARISON expr                     # comparison
    | 'sum' '(' expr ')'                       # allTimeSum
    | 'sum_connections' '(' portFieldExpr ')'  # portFieldSum
    | 'sum' '(' from=shift '..' to=shift ',' expr ')'  # timeSum
    | IDENTIFIER '(' argList? ')'              # function
    | IDENTIFIER '[' shift ']'                 # timeShift
    | IDENTIFIER '[' expr  ']'                 # timeIndex
    | '(' expr ')' '[' shift ']'               # timeShiftExpr
    | '(' expr ')' '[' expr ']'               # timeIndexExpr
    ;

argList : expr (',' expr)* ;

atom
    : NUMBER                                   # number
    | IDENTIFIER                               # identifier
    ;

// a shift is required to be either "t" or "t + ..." or "t - ..."
// Note: simply defining it as "shift: TIME ('+' | '-') expr" won't work
//       because the minus sign will not have the expected precedence:
//       "t - d + 1" would be equivalent to "t - (d + 1)"
shift: TIME shift_expr?;

// Because the shift MUST start with + or -, we need
// to differentiate it from generic "expr".
// A shift expression can only be extended to the right by a
// "right_expr" which cannot start with a + or -,
// unlike shift_expr itself.
//
// Unlike "expr", which is one left-recursive rule where precedence falls out
// of the order of the alternatives, this sub-grammar is a hand-written cascade:
// precedence is encoded in which rule each operand recurses into. An operator
// therefore has to be given its precedence twice, once in "expr" and once here,
// and the two must agree — "x[t-2^2]" is expected to shift by the value of the
// expression "-2^2". Adding an operator to "expr" alone gives it nothing inside
// a shift.
//
// TODO: the grammar is still a little weird, because we
//       allow more things in the "expr" parts of those
//       shift expressions than on their left-most part
//       (port fields, nested time shifts and so on).
shift_expr
    : shift_expr op=('*' | '/') right_expr     # shiftMuldiv
    | shift_expr op=('+' | '-') right_expr     # shiftAddsub
    | op=('+' | '-') shift_operand             # signedOperand
    ;

right_expr
    : right_expr op=('/' | '*') right_expr     # rightMuldiv
    | shift_operand                            # rightOperand
    ;

// The two highest precedence tiers of the shift sub-grammar, mirroring the
// "power" and atom levels of "expr". They are named after their level, not
// after '^': every operand of a shift goes through them, whether or not a
// power is involved.
//
// They must stay separate from "right_expr" for two reasons, both of them
// restating here what "expr" gets for free from its alternative order:
//  - the leading sign of "shift_expr" needs a power-capable operand, so that
//    '^' binds tighter than the sign and "t - 2^2" shifts by -4, matching
//    "-2^2" = "-(2^2)". That operand must NOT also swallow '*' and '/', or
//    "t - 2*3" becomes ambiguous and reassociates to "-(2*3)" instead of
//    "(-2)*3";
//  - "right_expr" is the muldiv tier, so on the right of '^' it would be
//    entered at precedence 0 and greedily swallow '*' and '/': "t - 2^2*3"
//    would mean "2^(2*3)" instead of "(2^2)*3".
//
// The one deliberate divergence from "expr" is that the exponent is unsigned
// here, so "t + 2^-1" is a parse error: a fractional time shift is
// meaningless, while negative exponents remain available in "expr".
shift_operand
    : shift_primary '^' shift_operand          # rightPower
    | shift_primary                            # rightPrimary
    ;

shift_primary
    : '(' expr ')'                             # rightExpression
    | atom                                     # rightAtom
    ;


fragment DIGIT         : [0-9] ;
fragment CHAR          : [a-zA-Z_];
fragment CHAR_OR_DIGIT : (CHAR | DIGIT);

NUMBER        : DIGIT+ ('.' DIGIT+)?;
TIME          : 't';
IDENTIFIER    : CHAR CHAR_OR_DIGIT*;
COMPARISON    : ( '=' | '>=' | '<=' );

WS: (' ' | '\t' | '\r'| '\n') -> skip;
