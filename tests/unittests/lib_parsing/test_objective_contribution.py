# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

"""Tests for objective creation logic and coefficient accumulation."""
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import ortools.linear_solver.pywraplp as lp
import pytest

# --- Mocks for External Dependencies ---


class TermKey:
    """Minimal mock for the TermKey."""

    def __init__(self, component_id: str, variable_name: str, scenario: int = 0):
        self.component_id = component_id
        self.variable_name = variable_name
        self.scenario = scenario

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TermKey):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((self.component_id, self.variable_name, self.scenario))


class Term:
    """Minimal mock for a single term in LinearExpression."""

    def __init__(self, key: TermKey, coefficient: float):
        self.key = key
        self.coefficient = coefficient


class LinearExpression:
    """Minimal mock for a linearized expression."""

    def __init__(self, constant: float, terms: Dict[TermKey, Term]):
        self.constant = constant
        self.terms = terms


class Component:
    """Minimal mock for a component."""

    def __init__(
        self,
        id: str,
        objective_contributions: Optional[Dict[str, LinearExpression]] = None,
    ):
        self.id = id
        self.model = Mock()
        self.model.objective_contributions = objective_contributions


class ComponentVariableKey:
    """Minimal mock for variable key used by the context."""

    def __init__(
        self,
        component_id: str,
        variable_name: str,
        scenario: int = 0,
        block_timestep: int = 0,
    ):
        self.component_id = component_id
        self.variable_name = variable_name
        self.scenario = scenario
        self.block_timestep = block_timestep

    def name(self) -> str:
        return f"{self.component_id}_{self.variable_name}_s{self.scenario}_t{self.block_timestep}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentVariableKey):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(self.name())


class MockSolverVariable:
    """Mocks an lp.Variable wrapper for the context."""

    def __init__(self, name: str, real_lp_var: lp.Variable):
        self._name = name
        self.is_in_objective = False
        self.mock_lp_var = real_lp_var

    def name(self) -> str:
        return self._name


def _setup_mock_optimization_environment(
    linear_expressions: Dict[Any, LinearExpression]
) -> Any:
    """Sets up a mock context and problem with real OR-Tools solver objects."""

    mock_problem = Mock()
    mock_problem.solver = lp.Solver.CreateSolver("GLOP")
    solver = mock_problem.solver

    mock_lp_vars: Dict[str, lp.Variable] = {}
    mock_context_vars: Dict[str, MockSolverVariable] = {}

    all_term_keys = set()
    for linear_expr in linear_expressions.values():
        if isinstance(linear_expr, LinearExpression):
            all_term_keys.update(linear_expr.terms.keys())

    for term_key in all_term_keys:
        comp_var_key = ComponentVariableKey(
            term_key.component_id, term_key.variable_name, term_key.scenario
        )
        name = comp_var_key.name()

        real_lp_var = solver.NumVar(0, solver.infinity(), name)
        mock_lp_vars[name] = real_lp_var

        mock_context_vars[name] = MockSolverVariable(name, real_lp_var)

    opt_context = Mock()
    opt_context._solver_variables = mock_context_vars
    mock_problem.context = opt_context

    def mock_get_solver_var(term: Term, context: Any) -> lp.Variable:
        key = ComponentVariableKey(
            term.key.component_id, term.key.variable_name, term.key.scenario
        )
        return mock_lp_vars[key.name()]

    opt_context._mock_get_solver_var = mock_get_solver_var

    linear_expressions_copy = linear_expressions.copy()

    def mock_linearize_expression(expanded_expr: Any) -> LinearExpression:
        if expanded_expr in linear_expressions_copy:
            return linear_expressions_copy.pop(expanded_expr)

        if isinstance(expanded_expr, Mock):
            key_to_pop = next(
                k
                for k, v in linear_expressions_copy.items()
                if isinstance(k, Mock) and k == expanded_expr
            )
            return linear_expressions_copy.pop(key_to_pop)

        raise ValueError(
            f"Expression {expanded_expr} not found for mock linearization."
        )

    opt_context.linearize_expression.side_effect = mock_linearize_expression

    return mock_problem, opt_context, mock_lp_vars, mock_get_solver_var


# --- Helper Functions (Mimicking OptimizationContext logic) ---


@patch(
    "gems.simulation.optimization._instantiate_model_expression",
    new=Mock(side_effect=lambda expr, *args: expr),
)
def _create_objective(
    solver: lp.Solver,
    opt_context: Any,
    component: Component,
    objective_contribution: LinearExpression,
) -> None:
    """Helper to create a single objective from a LinearExpression."""
    instantiated_expr = objective_contribution
    opt_context.expand_operators.return_value = instantiated_expr
    expanded = opt_context.expand_operators(instantiated_expr)
    linear_expr = opt_context.linearize_expression(expanded)

    with patch(
        "gems.simulation.optimization._get_solver_var",
        new=opt_context._mock_get_solver_var,
    ):
        _add_linear_expression_to_objective(solver, opt_context, linear_expr)


def _add_linear_expression_to_objective(
    solver: lp.Solver,
    context: Any,
    linear_expr: LinearExpression,
    existing_obj: Optional[lp.Objective] = None,
) -> lp.Objective:
    """Helper to accumulate a LinearExpression to the objective."""
    obj: lp.Objective = existing_obj if existing_obj is not None else solver.Objective()

    def _get_solver_var(term: Term, ctx: Any) -> lp.Variable:
        return context._mock_get_solver_var(term, ctx)

    for term in linear_expr.terms.values():
        solver_var = _get_solver_var(term, context)

        obj.SetCoefficient(
            solver_var,
            obj.GetCoefficient(solver_var) + term.coefficient,
        )

        context._solver_variables[solver_var.name()].is_in_objective = True

    obj.SetOffset(linear_expr.constant + obj.offset())
    return obj


# ==============================================================================
# --- Tests Start Here ---
# ==============================================================================


@patch(
    "gems.simulation.optimization._add_linear_expression_to_objective",
    new=_add_linear_expression_to_objective,
)
class TestObjectiveContributions:
    def test_objective_contribution_value_accumulation(self) -> None:
        """
        Verifies that multiple objective contributions correctly accumulate into
        a single OR-Tools objective.

        This test compares:
        1. A 'legacy' objective with a constant (20.0) and a single variable coefficient (10.0).
        2. A 'contribution' objective formed by summing two contributions (A: 7.0/3.0, B: 13.0/7.0).
        In other words:
        1. 10x + 20
        2. 3x + 7
           7x + 13
        It asserts that the accumulated objective has the doubled values (20.0 constant, 10.0 coeff),
        demonstrating correct merging logic for both offsets and coefficients.
        """
        comp_id = "test_comp"
        var_name = "X_invest"
        term_key = TermKey(comp_id, var_name)

        # --- LEGACY SETUP (10.0 constant, 5.0 coeff) ---
        legacy_expr = LinearExpression(
            constant=20.0, terms={term_key: Term(term_key, 10.0)}
        )

        # --- CONTRIBUTION SETUP (Total 20.0 constant, 10.0 coeff) ---
        contrib_a_mock = Mock()
        contrib_b_mock = Mock()
        contrib_a = LinearExpression(
            constant=7.0, terms={term_key: Term(term_key, 3.0)}
        )
        contrib_b = LinearExpression(
            constant=13.0, terms={term_key: Term(term_key, 7.0)}
        )

        legacy_comp = Component(comp_id, objective_contributions=None)
        contrib_comp = Component(
            comp_id,
            objective_contributions={
                "contrib_a": contrib_a_mock,
                "contrib_b": contrib_b_mock,
            },
        )

        legacy_mock_expressions = {legacy_expr: legacy_expr}
        legacy_problem, legacy_context, _, _ = _setup_mock_optimization_environment(
            legacy_mock_expressions
        )

        contrib_mock_expressions = {
            contrib_a_mock: contrib_a,
            contrib_b_mock: contrib_b,
        }
        contrib_problem, contrib_context, _, _ = _setup_mock_optimization_environment(
            contrib_mock_expressions
        )

        def run_legacy_creation():
            legacy_context.build_strategy.get_objectives.return_value = [legacy_expr]
            _create_objective(
                legacy_problem.solver, legacy_context, legacy_comp, legacy_expr
            )
            return legacy_problem.solver.Objective()

        def run_contrib_creation(problem: Any, context: Any, component: Component):
            main_objective: Optional[lp.Objective] = None
            for _, expr in component.model.objective_contributions.items():
                instantiated = expr
                context.expand_operators.return_value = instantiated
                expanded = context.expand_operators(instantiated)
                linear_expr = context.linearize_expression(expanded)

                with patch(
                    "gems.simulation.optimization._get_solver_var",
                    new=context._mock_get_solver_var,
                ):
                    main_objective = _add_linear_expression_to_objective(
                        problem.solver,
                        context,
                        linear_expr,
                        existing_obj=main_objective,
                    )
            return main_objective

        # --- EXECUTE ---
        legacy_obj = run_legacy_creation()
        contrib_obj = run_contrib_creation(
            contrib_problem, contrib_context, contrib_comp
        )

        # --- ASSERTIONS ---
        assert contrib_obj is not None

        # 1. Check Constant Offset
        assert legacy_obj.offset() == pytest.approx(20.0)
        assert contrib_obj.offset() == pytest.approx(20.0)

        # 2. Check Variable Coefficient
        solver_var_name = ComponentVariableKey(comp_id, var_name).name()

        # Get the correct solver variable instances for each environment
        legacy_solver_var = legacy_context._solver_variables[
            solver_var_name
        ].mock_lp_var
        contrib_solver_var = contrib_context._solver_variables[
            solver_var_name
        ].mock_lp_var

        assert legacy_obj.GetCoefficient(legacy_solver_var) == pytest.approx(10.0)
        assert contrib_obj.GetCoefficient(contrib_solver_var) == pytest.approx(10.0)

    def test_investment_single_continuous_candidate(self) -> None:
        """
        Verifies correct accumulation of production costs and investment costs for
        a single continuous investment candidate.

        The test combines two contributions:
        - prod_cost: terms for P_gen_a (45.0) and P_unsup_a (501.0).
        - invest_cost: terms for I_cand_cont (400.0) and P_cand_cont (10.0).

        It asserts that the final objective contains all four unique terms with
        the expected coefficients and a zero offset.
        """
        comp_id = "node_a"

        term_gen = TermKey(comp_id, "P_gen_a")
        term_unsup = TermKey(comp_id, "P_unsup_a")
        term_invest_cand = TermKey(comp_id, "I_cand_cont")
        term_prod_cand = TermKey(comp_id, "P_cand_cont")

        # --- Contributions as Linear Expressions ---
        prod_cost_expr = LinearExpression(
            constant=0.0,
            terms={
                term_gen: Term(term_gen, 45.0),
                term_unsup: Term(term_unsup, 501.0),
            },
        )

        invest_cost_expr = LinearExpression(
            constant=0.0,
            terms={
                term_invest_cand: Term(term_invest_cand, 400.0),
                term_prod_cand: Term(term_prod_cand, 10.0),
            },
        )

        test_comp = Component(
            comp_id,
            objective_contributions={
                "prod_cost": prod_cost_expr,
                "invest_cost": invest_cost_expr,
            },
        )

        mock_expressions = {
            prod_cost_expr: prod_cost_expr,
            invest_cost_expr: invest_cost_expr,
        }

        problem, context, _, _ = _setup_mock_optimization_environment(mock_expressions)
        context.expand_operators.side_effect = lambda expr: expr

        main_objective: Optional[lp.Objective] = None
        for _, expr in test_comp.model.objective_contributions.items():
            linear_expr = context.linearize_expression(expr)

            with patch(
                "gems.simulation.optimization._get_solver_var",
                new=context._mock_get_solver_var,
            ):
                main_objective = _add_linear_expression_to_objective(
                    problem.solver, context, linear_expr, existing_obj=main_objective
                )

        # --- ASSERTIONS ---
        assert main_objective is not None
        assert main_objective.offset() == pytest.approx(0.0)

        def get_coeff(term_key: TermKey) -> float:
            solver_var_name = ComponentVariableKey(
                term_key.component_id, term_key.variable_name
            ).name()
            solver_var = context._solver_variables[solver_var_name].mock_lp_var
            return main_objective.GetCoefficient(solver_var)

        assert get_coeff(term_gen) == pytest.approx(45.0)
        assert get_coeff(term_unsup) == pytest.approx(501.0)
        assert get_coeff(term_invest_cand) == pytest.approx(400.0)
        assert get_coeff(term_prod_cand) == pytest.approx(10.0)

    def test_investment_two_candidates_discrete(self) -> None:
        """
        Verifies the accumulation of costs involving two different investment candidates
        (one continuous, one discrete), resulting from three separate contributions.

        The test combines three contributions:
        - prod_cost: terms for P_gen_b (45.0) and P_unsup_b (501.0).
        - cont_cost: terms for continuous investment (490.0) and continuous production (10.0).
        - disc_cost: terms for discrete investment (200.0) and discrete production (10.0).

        It asserts that the final objective correctly contains coefficients for all six
        unique optimization variables.
        """
        comp_id = "node_b"

        term_gen = TermKey(comp_id, "P_gen_b")
        term_unsup = TermKey(comp_id, "P_unsup_b")
        term_invest_cont = TermKey(comp_id, "I_cand_cont")
        term_prod_cont = TermKey(comp_id, "P_cand_cont")
        term_invest_disc = TermKey(comp_id, "I_cand_disc")
        term_prod_disc = TermKey(comp_id, "P_cand_disc")

        # --- Contributions as Linear Expressions ---
        prod_cost_expr = LinearExpression(
            constant=0.0,
            terms={
                term_gen: Term(term_gen, 45.0),
                term_unsup: Term(term_unsup, 501.0),
            },
        )

        cont_cost_expr = LinearExpression(
            constant=0.0,
            terms={
                term_invest_cont: Term(term_invest_cont, 490.0),
                term_prod_cont: Term(term_prod_cont, 10.0),
            },
        )

        disc_cost_expr = LinearExpression(
            constant=0.0,
            terms={
                term_invest_disc: Term(term_invest_disc, 200.0),
                term_prod_disc: Term(term_prod_disc, 10.0),
            },
        )

        test_comp = Component(
            comp_id,
            objective_contributions={
                "prod_cost": prod_cost_expr,
                "cont_cost": cont_cost_expr,
                "disc_cost": disc_cost_expr,
            },
        )

        mock_expressions = {
            prod_cost_expr: prod_cost_expr,
            cont_cost_expr: cont_cost_expr,
            disc_cost_expr: disc_cost_expr,
        }

        problem, context, _, _ = _setup_mock_optimization_environment(mock_expressions)
        context.expand_operators.side_effect = lambda expr: expr

        main_objective: Optional[lp.Objective] = None
        for _, expr in test_comp.model.objective_contributions.items():
            linear_expr = context.linearize_expression(expr)

            with patch(
                "gems.simulation.optimization._get_solver_var",
                new=context._mock_get_solver_var,
            ):
                main_objective = _add_linear_expression_to_objective(
                    problem.solver, context, linear_expr, existing_obj=main_objective
                )

        # --- ASSERTIONS ---
        assert main_objective is not None

        def get_coeff(term_key: TermKey) -> float:
            solver_var_name = ComponentVariableKey(
                term_key.component_id, term_key.variable_name
            ).name()
            solver_var = context._solver_variables[solver_var_name].mock_lp_var
            return main_objective.GetCoefficient(solver_var)

        assert get_coeff(term_gen) == pytest.approx(45.0)
        assert get_coeff(term_unsup) == pytest.approx(501.0)
        assert get_coeff(term_invest_cont) == pytest.approx(490.0)
        assert get_coeff(term_prod_cont) == pytest.approx(10.0)
        assert get_coeff(term_invest_disc) == pytest.approx(200.0)
        assert get_coeff(term_prod_disc) == pytest.approx(10.0)
