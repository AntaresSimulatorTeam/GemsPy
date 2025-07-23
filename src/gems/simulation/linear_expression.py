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

"""
Specific modelling for "instantiated" linear expressions,
with only variables and literal coefficients.
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TypeVar, Union

from gems.expression.expression import (
    OneScenarioIndex,
    ScenarioIndex,
    TimeIndex,
    TimeShift,
    TimeStep,
)

T = TypeVar("T")

EPS = 10 ** (-16)


def is_close_abs(value: float, other_value: float, eps: float) -> bool:
    return abs(value - other_value) < eps


def is_zero(value: float) -> bool:
    return is_close_abs(value, 0, EPS)


def is_one(value: float) -> bool:
    return is_close_abs(value, 1, EPS)


def is_minus_one(value: float) -> bool:
    return is_close_abs(value, -1, EPS)


@dataclass(frozen=True)
class TimeExpansion:
    """
    Carries knowledge of which timesteps this term refers to.
    Simplest one is "only the current timestep"
    """

    def get_timesteps(self, current_timestep: int, block_length: int) -> List[int]:
        return [current_timestep]

    def apply(self, other: "TimeExpansion") -> "TimeExpansion":
        """
        Apply another time expansion on this one.
        For example, a shift of -1 applied to a shift one +1 could provide
        a no-op TimeExpansion. Not yet supported for now, though.
        """
        return other


@dataclass(frozen=True)
class AllTimeExpansion(TimeExpansion):
    def get_timesteps(self, current_timestep: int, block_length: int) -> List[int]:
        return [t for t in range(block_length)]

    def apply(self, other: "TimeExpansion") -> "TimeExpansion":
        raise ValueError("No time operation allowed on all-time sum.")


@dataclass(frozen=True)
class TimeEvalExpansion(TimeExpansion):
    timestep: int

    def get_timesteps(self, current_timestep: int, block_length: int) -> List[int]:
        return [self.timestep]

    def apply(self, other: "TimeExpansion") -> "TimeExpansion":
        raise ValueError(
            "Time operation on evaluated expression not supported for now."
        )


@dataclass(frozen=True)
class TimeShiftExpansion(TimeExpansion):
    shift: int

    def get_timesteps(self, current_timestep: int, block_length: int) -> List[int]:
        return [current_timestep + self.shift]

    def apply(self, other: "TimeExpansion") -> "TimeExpansion":
        raise ValueError("Time operation on shifted expression not supported for now.")


@dataclass(frozen=True)
class TimeSumExpansion(TimeExpansion):
    from_shift: int
    to_shift: int

    def get_timesteps(self, current_timestep: int, block_length: int) -> List[int]:
        return [
            t
            for t in range(
                current_timestep + self.from_shift, current_timestep + self.to_shift
            )
        ]

    def apply(self, other: "TimeExpansion") -> "TimeExpansion":
        raise ValueError("Time operation on time-sums not supported for now.")


@dataclass(frozen=True)
class TermKey:

    """
    Utility class to provide key for a term that contains all term information except coefficient
    """

    component_id: str
    variable_name: str
    time_index: Optional[int]
    scenario_index: Optional[int]


def _str_for_coeff(coeff: float) -> str:
    if is_one(coeff):
        return "+"
    elif is_minus_one(coeff):
        return "-"
    else:
        return "{:+g}".format(coeff)


def _time_index_to_str(time_index: TimeIndex) -> str:
    if isinstance(time_index, TimeShift):
        if time_index.timeshift == 0:
            return "t"
        elif time_index.timeshift > 0:
            return f"t + {time_index.timeshift}"
        else:
            return f"t - {-time_index.timeshift}"
    if isinstance(time_index, TimeStep):
        return f"{time_index.timestep}"
    return ""


def _scenario_index_to_str(scenario_index: ScenarioIndex) -> str:
    if isinstance(scenario_index, OneScenarioIndex):
        return f"{scenario_index.scenario}"
    return ""


def _str_for_time_expansion(exp: TimeExpansion) -> str:
    if isinstance(exp, TimeShiftExpansion):
        return f".shift({exp.shift})"
    elif isinstance(exp, TimeSumExpansion):
        return f".sum({exp.from_shift}, {exp.to_shift})"
    elif isinstance(exp, AllTimeExpansion):
        return ".sum()"
    else:
        return ""


@dataclass(frozen=True)
class Term:
    """
    One term in a linear expression: for example the "10x" par in "10x + 5y + 5"

    Args:
        coefficient: the coefficient for that term, for example "10" in "10x"
        variable_name: the name of the variable, for example "x" in "10x"
    """

    coefficient: float
    component_id: str
    variable_name: str
    time_index: Optional[int]
    scenario_index: Optional[int]

    # TODO: It may be useful to define __add__, __sub__, etc on terms, which should return a linear expression ?

    def is_zero(self) -> bool:
        return is_zero(self.coefficient)

    def __str__(self) -> str:
        # Useful for debugging tests
        return repr(self)

    def __repr__(self) -> str:
        # Useful for debugging tests
        result = (
            f"{_str_for_coeff(self.coefficient)}{self.component_id}.{self.variable_name}"
            f"[{self.time_index},{self.scenario_index}]"
        )
        return result


def generate_key(term: Term) -> TermKey:
    return TermKey(
        term.component_id,
        term.variable_name,
        term.time_index,
        term.scenario_index,
    )


def _merge_dicts(
    lhs: Dict[TermKey, Term],
    rhs: Dict[TermKey, Term],
    merge_func: Callable[[Optional[Term], Optional[Term]], Term],
) -> Dict[TermKey, Term]:
    res = {}
    for k, left in lhs.items():
        right = rhs.get(k, None)
        res[k] = merge_func(left, right)
    for k, right in rhs.items():
        if k not in lhs:
            res[k] = merge_func(None, right)
    return res


def _add_terms(lhs: Optional[Term], rhs: Optional[Term]) -> Term:
    if lhs is not None and rhs is not None:
        return dataclasses.replace(rhs, coefficient=lhs.coefficient + rhs.coefficient)
    elif lhs is not None and rhs is None:
        return lhs
    elif lhs is None and rhs is not None:
        return rhs
    raise ValueError("Cannot add 2 null terms.")


def _substract_terms(lhs: Optional[Term], rhs: Optional[Term]) -> Term:
    if lhs is not None and rhs is not None:
        return dataclasses.replace(lhs, coefficient=lhs.coefficient - rhs.coefficient)
    elif lhs is not None and rhs is None:
        return lhs
    elif lhs is None and rhs is not None:
        return dataclasses.replace(rhs, coefficient=-rhs.coefficient)
    raise ValueError("Cannot subtract 2 null terms.")


class LinearExpression:
    """
    Represents a linear expression with respect to variable names, for example 10x + 5y + 2.

    Operators may be used for construction.

    Args:
        terms: the list of variable terms, for example 10x and 5y in "10x + 5y + 2".
        constant: the constant term, for example 2 in "10x + 5y + 2"

    Examples:
        Operators may be used for construction:

        >>> LinearExpression([], 10) + LinearExpression([Term(10, "x")], 0)
        LinearExpression([Term(10, "x")], 10)
    """

    terms: Dict[TermKey, Term]
    constant: float

    def __init__(
        self,
        terms: Optional[Union[Dict[TermKey, Term], List[Term]]] = None,
        constant: Optional[float] = None,
    ) -> None:
        self.constant = 0
        self.terms = {}

        if constant is not None:
            # += b
            self.constant = constant
        if terms is not None:
            # Allows to give two different syntax in the constructor:
            #   - List[Term] is natural
            #   - Dict[str, Term] is useful when constructing a linear expression from the terms of another expression
            if isinstance(terms, dict):
                for term_key, term in terms.items():
                    if not term.is_zero():
                        self.terms[term_key] = term
            elif isinstance(terms, list):
                for term in terms:
                    if not term.is_zero():
                        self.terms[generate_key(term)] = term
            else:
                raise TypeError(
                    f"Terms must be either of type Dict[str, Term] or List[Term], whereas {terms} is of type {type(terms)}"
                )

    def is_zero(self) -> bool:
        return len(self.terms) == 0 and is_zero(self.constant)

    def str_for_constant(self) -> str:
        if is_zero(self.constant):
            return ""
        else:
            return "{:+g}".format(self.constant)

    def __repr__(self) -> str:
        # Useful for debugging tests
        result = ""
        if self.is_zero():
            result += "0"
        else:
            for term in self.terms.values():
                result += repr(term)

            result += self.str_for_constant()

        return result

    def __eq__(self, rhs: object) -> bool:
        return (
            isinstance(rhs, LinearExpression)
            and is_close_abs(self.constant, rhs.constant, EPS)
            and self.terms
            == rhs.terms  # /!\ There may be float equality comparison in the terms values
        )

    def __iadd__(self, rhs: "LinearExpression") -> "LinearExpression":
        if not isinstance(rhs, LinearExpression):
            return NotImplemented
        self.constant += rhs.constant
        aggregated_terms = _merge_dicts(self.terms, rhs.terms, _add_terms)
        self.terms = aggregated_terms
        self.remove_zeros_from_terms()
        return self

    def __add__(self, rhs: "LinearExpression") -> "LinearExpression":
        result = LinearExpression()
        result += self
        result += rhs
        return result

    def __isub__(self, rhs: "LinearExpression") -> "LinearExpression":
        if not isinstance(rhs, LinearExpression):
            return NotImplemented
        self.constant -= rhs.constant
        aggregated_terms = _merge_dicts(self.terms, rhs.terms, _substract_terms)
        self.terms = aggregated_terms
        self.remove_zeros_from_terms()
        return self

    def __sub__(self, rhs: "LinearExpression") -> "LinearExpression":
        result = LinearExpression()
        result += self
        result -= rhs
        return result

    def __neg__(self) -> "LinearExpression":
        result = LinearExpression()
        result -= self
        return result

    def __imul__(self, rhs: "LinearExpression") -> "LinearExpression":
        if not isinstance(rhs, LinearExpression):
            return NotImplemented

        if self.terms and rhs.terms:
            raise ValueError("Cannot multiply two non constant expression")
        else:
            if self.terms:
                left_expr = self
                const_expr = rhs
            else:
                # It is possible that both expr are constant
                left_expr = rhs
                const_expr = self
            if is_close_abs(const_expr.constant, 0, EPS):
                return LinearExpression()
            elif is_close_abs(const_expr.constant, 1, EPS):
                _copy_expression(left_expr, self)
            else:
                left_expr.constant *= const_expr.constant
                for term_key, term in left_expr.terms.items():
                    left_expr.terms[term_key] = Term(
                        term.coefficient * const_expr.constant,
                        term.component_id,
                        term.variable_name,
                        term.time_index,
                        term.scenario_index,
                    )
                _copy_expression(left_expr, self)
        return self

    def __mul__(self, rhs: "LinearExpression") -> "LinearExpression":
        result = LinearExpression()
        result += self
        result *= rhs
        return result

    def __itruediv__(self, rhs: "LinearExpression") -> "LinearExpression":
        if not isinstance(rhs, LinearExpression):
            return NotImplemented

        if rhs.terms:
            raise ValueError("Cannot divide by a non constant expression")
        else:
            if is_close_abs(rhs.constant, 0, EPS):
                raise ZeroDivisionError("Cannot divide expression by zero")
            elif is_close_abs(rhs.constant, 1, EPS):
                return self
            else:
                self.constant /= rhs.constant
                for term_key, term in self.terms.items():
                    self.terms[term_key] = Term(
                        term.coefficient / rhs.constant,
                        term.component_id,
                        term.variable_name,
                        term.time_index,
                        term.scenario_index,
                    )
        return self

    def __truediv__(self, rhs: "LinearExpression") -> "LinearExpression":
        result = LinearExpression()
        result += self
        result /= rhs

        return result

    def remove_zeros_from_terms(self) -> None:
        # TODO: Not optimized, checks could be done directly when doing operations on self.linear_term to avoid copies
        for term_key, term in self.terms.copy().items():
            if is_close_abs(term.coefficient, 0, EPS):
                del self.terms[term_key]


def _copy_expression(src: LinearExpression, dst: LinearExpression) -> None:
    dst.terms = src.terms
    dst.constant = src.constant
