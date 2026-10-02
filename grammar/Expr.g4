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
    | '-' expr                                 # negation
    | '(' expr ')'                             # expression
    | expr op=('/' | '*') expr                 # muldiv
    | expr op=('+' | '-') expr                 # addsub
    | expr COMPARISON expr                     # comparison
    | 'sum' '(' expr ')'                       # allTimeSum
    | 'sum_connections' '(' portFieldExpr ')'  # portFieldSum
    | 'sum' '(' from=shift '..' to=shift ',' expr ')'  # timeSum
    | 'sum_over' '(' IDENTIFIER ',' expr ')'   # sumOver
    | IDENTIFIER '(' argList? ')'              # function
    | IDENTIFIER '[' indexList ']'             # bracketIndex
    | '(' expr ')' '[' indexList ']'           # bracketIndexExpr
    ;

argList : expr (',' expr)* ;

// An index list is one or more comma-separated terms, in any order.
// ANTLR only distinguishes the three shapes below; it cannot tell whether a
// bare/signed identifier is a declared set, "t", or an ordinary
// parameter -- that's resolved by ExpressionNodeBuilderVisitor via
// ModelIdentifiers. Term order not mattering (`X[fuel=3, 2]` == `X[2,
// fuel=3]`) and rejecting more than one term denoting the time dimension
// (`X[2, 3]`) are also builder-level checks, not grammar-level ones.
indexList : indexTerm (',' indexTerm)* ;

// keywordTerm reuses the COMPARISON token rather than a bare '=' literal,
// which would otherwise create a second, competing implicit lexer token for
// '=' and shadow COMPARISON everywhere else. The builder rejects anything
// other than a literal '=' here.
indexTerm
    : shift                                    # namedOrTimeShiftTerm
    | (TIME | IDENTIFIER) COMPARISON expr      # keywordTerm
    | expr                                     # positionTerm
    ;

atom
    : NUMBER                                   # number
    | IDENTIFIER                               # identifier
    ;

// A shift is required to be either "t"/a set id, or "t + ..."/"t - ...", or
// "<set id> + ..."/"<set id> - ...".
// Note: simply defining it as "shift: (TIME|IDENTIFIER) ('+' | '-') expr"
//       won't work because the minus sign will not have the expected
//       precedence: "t - d + 1" would be equivalent to "t - (d + 1)"
shift: (TIME | IDENTIFIER) shift_expr?;

// Because the shift MUST start with + or -, we need
// to differentiate it from generic "expr".
// A shift expression can only be extended to the right by a
// "right_expr" which cannot start with a + or -,
// unlike shift_expr itself.
// TODO: the grammar is still a little weird, because we
//       allow more things in the "expr" parts of those
//       shift expressions than on their left-most part
//       (port fields, nested time shifts and so on).
shift_expr
    : shift_expr op=('*' | '/') right_expr     # shiftMuldiv
    | shift_expr op=('+' | '-') right_expr     # shiftAddsub
    | op=('+' | '-') atom                      # signedAtom
    | op=('+' | '-') '(' expr ')'              # signedExpression
    ;

right_expr
    : right_expr op=('/' | '*') right_expr     # rightMuldiv
    | '(' expr ')'                             # rightExpression
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
