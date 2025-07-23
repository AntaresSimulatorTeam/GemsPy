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
The optimization module contains the logic to translate the input model
into a mathematical optimization problem.
"""

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional

import ortools.linear_solver.pywraplp as lp

from gems.expression import EvaluationVisitor, ExpressionNode, ValueProvider, visit
from gems.expression.context_adder import add_component_context
from gems.expression.indexing import IndexingStructureProvider, compute_indexation
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.operators_expansion import ProblemDimensions, expand_operators
from gems.expression.port_resolver import PortFieldKey, resolve_port
from gems.model.common import ValueType
from gems.model.constraint import Constraint
from gems.model.port import PortFieldId
from gems.simulation.linear_expression import LinearExpression, Term
from gems.simulation.linearize import ParameterGetter, linearize_expression
from gems.simulation.strategy import (
    MergedProblemStrategy,
    ModelSelectionStrategy,
    RiskManagementStrategy,
    UniformRisk,
)
from gems.simulation.time_block import TimeBlock
from gems.study.data import DataBase
from gems.study.network import Component, Network
from gems.utils import get_or_add


@dataclass(eq=True, frozen=True)
class TimestepComponentVariableKey:
    """
    Identifies the solver variable for one timestep and one component variable.
    """

    component_id: str
    variable_name: str
    block_timestep: Optional[int] = None
    scenario: Optional[int] = None


def _get_parameter_value(
    context: "OptimizationContext",
    block_timestep: Optional[int],
    scenario: Optional[int],
    component_id: str,
    name: str,
) -> float:
    data = context.database.get_data(component_id, name)
    absolute_timestep = context.block_timestep_to_absolute_timestep(block_timestep)
    return data.get_value(absolute_timestep, scenario, context.tree_node)


def _make_value_provider(
    context: "OptimizationContext",
    block_timestep: Optional[int],
    scenario: Optional[int],
) -> ValueProvider:
    """
    Create a value provider which takes its values from
    the parameter values as defined in the network data.

    Cannot evaluate expressions which contain variables.
    """

    class Impl(ValueProvider):
        def get_component_variable_value(self, component_id: str, name: str) -> float:
            raise NotImplementedError(
                "Cannot provide variable value at problem build time."
            )

        def get_component_parameter_value(self, component_id: str, name: str) -> float:
            return _get_parameter_value(
                context, block_timestep, scenario, component_id, name
            )

        def get_variable_value(self, name: str) -> float:
            raise NotImplementedError(
                "Cannot provide variable value at problem build time."
            )

        def get_parameter_value(self, name: str) -> float:
            raise ValueError(
                "Parameter must be associated to its component before resolution."
            )

    return Impl()


def _compute_expression_value(
    expression: ExpressionNode,
    context: "OptimizationContext",
    block_timestep: Optional[int],
    scenario: Optional[int],
) -> float:
    value_provider = _make_value_provider(context, block_timestep, scenario)
    visitor = EvaluationVisitor(value_provider)
    return visit(expression, visitor)


class BlockBorderManagement(Enum):
    """
    Class to specify the way of handling the time horizon (or time block) border.
        - IGNORE_OUT_OF_FRAME: Ignore terms in constraints that lead to out of horizon data
        - CYCLE: Consider all timesteps to be specified modulo the horizon length, this is the actual functioning of Antares
    """

    IGNORE_OUT_OF_FRAME = "IGNORE"
    CYCLE = "CYCLE"


@dataclass
class SolverVariableInfo:
    """
    Helper class for constructing the structure file
    for Benders solver. It keeps track of the corresponding
    column of the variable in the MPS format as well as if it is
    present in the objective function or not
    """

    name: str
    column_id: int
    is_in_objective: bool


def float_to_int(value: float) -> int:
    if isinstance(value, int) or value.is_integer():
        return int(value)
    else:
        raise ValueError(f"{value} is not an integer.")


class OptimizationContext:
    """
    Helper class to build the optimization problem.

     - Maintains mappings between model and solver objects,
     - Maintains mappings between port fields and expressions,
     - Provides implementations of interfaces required by various visitors
       used to transform expressions (values providers ...).
    """

    def __init__(
        self,
        network: Network,
        database: DataBase,
        block: TimeBlock,
        scenarios: int,
        border_management: BlockBorderManagement,
        build_strategy: ModelSelectionStrategy = MergedProblemStrategy(),
        risk_strategy: RiskManagementStrategy = UniformRisk(),
        decision_tree_node: str = "",
        use_full_var_name: bool = True,
    ):
        self._network = network
        self._database = database
        self._block = block
        self._scenarios = scenarios
        self._border_management = border_management
        self._build_strategy = build_strategy
        self._risk_strategy = risk_strategy
        self._tree_node = decision_tree_node
        self._full_var_name = use_full_var_name

        self._component_variables: Dict[TimestepComponentVariableKey, lp.Variable] = {}
        self._solver_variables: Dict[str, SolverVariableInfo] = {}
        self._connection_fields_expressions: Dict[
            PortFieldKey, List[ExpressionNode]
        ] = {}

        self._constant_value_provider = self._make_constant_value_provider()
        self._indexing_structure_provider = self._make_data_structure_provider()
        self._parameter_getter = self._make_parameter_getter()

    @property
    def network(self) -> Network:
        return self._network

    @property
    def scenarios(self) -> int:
        return self._scenarios

    @property
    def tree_node(self) -> str:
        return self._tree_node

    @property
    def build_strategy(self) -> ModelSelectionStrategy:
        return self._build_strategy

    @property
    def risk_strategy(self) -> RiskManagementStrategy:
        return self._risk_strategy

    @property
    def full_var_name(self) -> bool:
        return self._full_var_name

    def block_length(self) -> int:
        return len(self._block.timesteps)

    @property
    def connection_fields_expressions(self) -> Dict[PortFieldKey, List[ExpressionNode]]:
        return self._connection_fields_expressions

    # TODO: Need to think about data processing when creating blocks with varying or inequal time steps length (aggregation, sum ?, mean of data ?)
    def block_timestep_to_absolute_timestep(
        self, block_timestep: Optional[int]
    ) -> Optional[int]:
        """
        Timestep may be None for parameters or variables that don't depend on time.
        """
        if block_timestep is None:
            return None
        return self._block.timesteps[self.get_actual_block_timestep(block_timestep)]

    def get_actual_block_timestep(self, block_timestep: int) -> int:
        if self._border_management == BlockBorderManagement.CYCLE:
            return block_timestep % self.block_length()
        else:
            raise NotImplementedError()

    @property
    def database(self) -> DataBase:
        return self._database

    def _manage_border_timesteps(self, timestep: int) -> int:
        if self._border_management == BlockBorderManagement.CYCLE:
            return timestep % self.block_length()
        else:
            raise NotImplementedError

    def get_time_indices(self, index_structure: IndexingStructure) -> Iterable[int]:
        return range(self.block_length()) if index_structure.time else range(1)

    def get_scenario_indices(self, index_structure: IndexingStructure) -> Iterable[int]:
        return range(self.scenarios) if index_structure.scenario else range(1)

    def get_component_variable(
        self,
        block_timestep: Optional[int],
        scenario: Optional[int],
        component_id: str,
        variable_name: str,
    ) -> lp.Variable:
        if block_timestep is not None:
            block_timestep = self._manage_border_timesteps(block_timestep)
        return self._component_variables[
            TimestepComponentVariableKey(
                component_id, variable_name, block_timestep, scenario
            )
        ]

    def get_all_component_variables(
        self,
    ) -> Dict[TimestepComponentVariableKey, lp.Variable]:
        return self._component_variables

    def register_component_variable(
        self,
        block_timestep: Optional[int],
        scenario: Optional[int],
        component_id: str,
        model_var_name: str,
        variable: lp.Variable,
    ) -> None:
        key = TimestepComponentVariableKey(
            component_id, model_var_name, block_timestep, scenario
        )
        if key not in self._component_variables:
            self._solver_variables[variable.name()] = SolverVariableInfo(
                variable.name(), len(self._solver_variables), False
            )
        self._component_variables[key] = variable

    def register_connection_fields_expressions(
        self,
        component_id: str,
        port_name: str,
        field_name: str,
        expression: ExpressionNode,
    ) -> None:
        key = PortFieldKey(component_id, PortFieldId(port_name, field_name))
        get_or_add(self._connection_fields_expressions, key, lambda: []).append(
            expression
        )

    def evaluate_time_bound(self, expression: ExpressionNode) -> int:
        res = visit(expression, EvaluationVisitor(self._constant_value_provider))
        return float_to_int(res)

    def _make_data_structure_provider(self) -> IndexingStructureProvider:
        """
        Retrieve information in data structure (parameter and variable) from the model
        """
        network = self.network

        class Impl(IndexingStructureProvider):
            def get_component_variable_structure(
                self, component_id: str, name: str
            ) -> IndexingStructure:
                return (
                    network.get_component(component_id).model.variables[name].structure
                )

            def get_component_parameter_structure(
                self, component_id: str, name: str
            ) -> IndexingStructure:
                return (
                    network.get_component(component_id).model.parameters[name].structure
                )

            def get_parameter_structure(self, name: str) -> IndexingStructure:
                raise RuntimeError("Component context should have been initialized.")

            def get_variable_structure(self, name: str) -> IndexingStructure:
                raise RuntimeError("Component context should have been initialized.")

        return Impl()

    def expand_operators(self, expression: ExpressionNode) -> ExpressionNode:
        dimensions = ProblemDimensions(self.block_length(), self.scenarios)
        time_bound_evaluator = self.evaluate_time_bound
        return expand_operators(
            expression,
            dimensions,
            time_bound_evaluator,
            self._indexing_structure_provider,
        )

    def _make_parameter_getter(self) -> ParameterGetter:
        ctxt = self

        class Impl(ParameterGetter):
            def get_parameter_value(
                self,
                component_id: str,
                parameter_name: str,
                timestep: Optional[int],
                scenario: Optional[int],
            ) -> float:
                return _get_parameter_value(
                    ctxt,
                    timestep,
                    scenario,
                    component_id,
                    parameter_name,
                )

        return Impl()

    def linearize_expression(
        self,
        expanded: ExpressionNode,
        timestep: Optional[int] = None,
        scenario: Optional[int] = None,
    ) -> LinearExpression:
        return linearize_expression(
            expanded, timestep, scenario, self._parameter_getter
        )

    def compute_indexing(self, expression: ExpressionNode) -> IndexingStructure:
        return compute_indexation(expression, self._indexing_structure_provider)

    def _make_constant_value_provider(self) -> ValueProvider:
        """
        Value provider which only provides values for constant parameters
        """
        context = self
        network = self.network

        class Impl(ValueProvider):
            def get_component_variable_value(
                self, component_id: str, name: str
            ) -> float:
                raise NotImplementedError(
                    "Cannot provide variable value at problem build time."
                )

            def get_component_parameter_value(
                self, component_id: str, name: str
            ) -> float:
                model = network.get_component(component_id).model
                structure = model.parameters[name].structure
                if structure.time or structure.scenario:
                    raise ValueError(f"Parameter {name} is not constant.")
                return _get_parameter_value(context, None, None, component_id, name)

            def get_variable_value(self, name: str) -> float:
                raise NotImplementedError(
                    "Cannot provide variable value at problem build time."
                )

            def get_parameter_value(self, name: str) -> float:
                raise ValueError(
                    "Parameter must be associated to its component before resolution."
                )

        return Impl()


def _compute_indexing(
    context: OptimizationContext, constraint: Constraint
) -> IndexingStructure:
    return (
        context.compute_indexing(constraint.expression)
        or context.compute_indexing(constraint.lower_bound)
        or context.compute_indexing(constraint.upper_bound)
    )


def _instantiate_model_expression(
    model_expression: ExpressionNode,
    component_id: str,
    optimization_context: OptimizationContext,
) -> ExpressionNode:
    """
    Performs common operations that are necessary on model expressions before their actual use:
     1. add component ID for variables and parameters of THIS component
     2. replace port fields by their definition
    """
    with_component = add_component_context(component_id, model_expression)
    with_component_and_ports = resolve_port(
        with_component, component_id, optimization_context.connection_fields_expressions
    )
    return with_component_and_ports


def _create_constraint(
    solver: lp.Solver,
    context: OptimizationContext,
    constraint: Constraint,
) -> None:
    """
    Adds a component-related constraint to the solver.
    """
    expanded = context.expand_operators(constraint.expression)
    constraint_indexing = _compute_indexing(context, constraint)

    for block_timestep in context.get_time_indices(constraint_indexing):
        for scenario in context.get_scenario_indices(constraint_indexing):
            linear_expr_at_t = context.linearize_expression(
                expanded, block_timestep, scenario
            )

            # What happens if there is some time_operator in the bounds ?
            constraint_data = ConstraintData(
                name=constraint.name,
                lower_bound=_compute_expression_value(
                    constraint.lower_bound,
                    context,
                    block_timestep,
                    scenario,
                ),
                upper_bound=_compute_expression_value(
                    constraint.upper_bound,
                    context,
                    block_timestep,
                    scenario,
                ),
                expression=linear_expr_at_t,
            )
            make_constraint(
                solver,
                context,
                block_timestep,
                scenario,
                constraint_data,
            )


def _create_objective(
    solver: lp.Solver,
    opt_context: OptimizationContext,
    component: Component,
    objective_contribution: ExpressionNode,
) -> None:
    instantiated_expr = _instantiate_model_expression(
        objective_contribution, component.id, opt_context
    )
    expanded = opt_context.expand_operators(instantiated_expr)
    linear_expr = opt_context.linearize_expression(expanded)

    obj: lp.Objective = solver.Objective()
    for term in linear_expr.terms.values():
        solver_var = _get_solver_var(
            term,
            opt_context,
        )
        opt_context._solver_variables[solver_var.name()].is_in_objective = True
        obj.SetCoefficient(
            solver_var,
            obj.GetCoefficient(solver_var) + term.coefficient,
        )

    # This should have no effect on the optimization
    obj.SetOffset(linear_expr.constant + obj.offset())


@dataclass
class ConstraintData:
    name: str
    lower_bound: float
    upper_bound: float
    expression: LinearExpression


def _get_solver_var(
    term: Term,
    context: OptimizationContext,
) -> lp.Variable:
    return context.get_component_variable(
        term.time_index,
        term.scenario_index,
        term.component_id,
        term.variable_name,
    )


def make_constraint(
    solver: lp.Solver,
    context: OptimizationContext,
    block_timestep: int,
    scenario: int,
    data: ConstraintData,
) -> None:
    """
    Adds constraint to the solver.
    """
    constraint_name = f"{data.name}_t{block_timestep}_s{scenario}"

    solver_constraint: lp.Constraint = solver.Constraint(constraint_name)
    constant: float = 0
    for term in data.expression.terms.values():
        solver_var = _get_solver_var(
            term,
            context,
        )
        coefficient = term.coefficient + solver_constraint.GetCoefficient(solver_var)
        solver_constraint.SetCoefficient(solver_var, coefficient)

    constant += data.expression.constant
    solver_constraint.SetBounds(
        data.lower_bound - constant, data.upper_bound - constant
    )


class OptimizationProblem:
    name: str
    solver: lp.Solver
    context: OptimizationContext

    def __init__(
        self,
        name: str,
        solver: lp.Solver,
        opt_context: OptimizationContext,
    ) -> None:
        self.name = name
        self.solver = solver
        self.context = opt_context

        self._register_connection_fields_definitions()
        self._create_variables()
        self._create_constraints()
        self._create_objectives()

    def _register_connection_fields_definitions(self) -> None:
        for cnx in self.context.network.connections:
            for field_name in list(cnx.master_port.keys()):
                master_port = cnx.master_port[field_name]
                port_definition = (
                    master_port.component.model.port_fields_definitions.get(
                        PortFieldId(
                            port_name=master_port.port_id, field_name=field_name.name
                        )
                    )
                )
                expression_node = port_definition.definition  # type: ignore
                instantiated_expression = add_component_context(
                    master_port.component.id, expression_node
                )
                self.context.register_connection_fields_expressions(
                    component_id=cnx.port1.component.id,
                    port_name=cnx.port1.port_id,
                    field_name=field_name.name,
                    expression=instantiated_expression,
                )
                self.context.register_connection_fields_expressions(
                    component_id=cnx.port2.component.id,
                    port_name=cnx.port2.port_id,
                    field_name=field_name.name,
                    expression=instantiated_expression,
                )

    def _solver_variable_name(
        self, component_id: str, var_name: str, t: Optional[int], s: Optional[int]
    ) -> str:
        component_prefix = (
            f"{component_id}_" if (self.context.full_var_name and component_id) else ""
        )
        tree_prefix = (
            f"{self.context.tree_node}_"
            if (self.context.full_var_name and self.context.tree_node)
            else ""
        )
        scenario_suffix = (
            f"_s{s}" if (s is not None and self.context.scenarios > 1) else ""
        )
        block_suffix = (
            f"_t{t}" if (t is not None and self.context.block_length() > 1) else ""
        )

        # Set solver var name
        # Externally, for the Solver, this variable will have a full name
        # Internally, it will be indexed by a structure that into account
        # the component id, variable name, timestep and scenario separately
        return (
            f"{tree_prefix}{component_prefix}{var_name}{block_suffix}{scenario_suffix}"
        )

    def _create_variables(self) -> None:
        for component in self.context.network.all_components:
            model = component.model

            for model_var in self.context.build_strategy.get_variables(model):
                var_indexing = model_var.structure
                instantiated_lb_expr = None
                instantiated_ub_expr = None

                if model_var.lower_bound:
                    instantiated_lb_expr = _instantiate_model_expression(
                        model_var.lower_bound, component.id, self.context
                    )
                if model_var.upper_bound:
                    instantiated_ub_expr = _instantiate_model_expression(
                        model_var.upper_bound, component.id, self.context
                    )

                time_indices: Iterable[Optional[int]] = [None]
                if var_indexing.is_time_varying():
                    time_indices = self.context.get_time_indices(var_indexing)

                scenario_indices: Iterable[Optional[int]] = [None]
                if var_indexing.is_scenario_varying():
                    scenario_indices = self.context.get_scenario_indices(var_indexing)

                for t, s in itertools.product(time_indices, scenario_indices):
                    lower_bound = -self.solver.infinity()
                    upper_bound = self.solver.infinity()
                    if instantiated_lb_expr:
                        lower_bound = _compute_expression_value(
                            instantiated_lb_expr, self.context, t, s
                        )
                    if instantiated_ub_expr:
                        upper_bound = _compute_expression_value(
                            instantiated_ub_expr, self.context, t, s
                        )

                    solver_var_name = self._solver_variable_name(
                        component.id, model_var.name, t, s
                    )

                    if lower_bound > upper_bound:
                        raise ValueError(
                            f"Upper bound ({upper_bound}) must be strictly greater than lower bound ({lower_bound}) for variable {solver_var_name}"
                        )

                    if model_var.data_type == ValueType.BOOLEAN:
                        solver_var = self.solver.BoolVar(
                            solver_var_name,
                        )
                    elif model_var.data_type == ValueType.INTEGER:
                        solver_var = self.solver.IntVar(
                            lower_bound,
                            upper_bound,
                            solver_var_name,
                        )
                    else:
                        solver_var = self.solver.NumVar(
                            lower_bound,
                            upper_bound,
                            solver_var_name,
                        )
                    self.context.register_component_variable(
                        t, s, component.id, model_var.name, solver_var
                    )

    def _create_constraints(self) -> None:
        for component in self.context.network.all_components:
            for constraint in self.context.build_strategy.get_constraints(
                component.model
            ):
                instantiated_expr = _instantiate_model_expression(
                    constraint.expression, component.id, self.context
                )
                instantiated_lb = _instantiate_model_expression(
                    constraint.lower_bound, component.id, self.context
                )
                instantiated_ub = _instantiate_model_expression(
                    constraint.upper_bound, component.id, self.context
                )

                instantiated_constraint = Constraint(
                    name=f"{component.id}_{constraint.name}",
                    expression=instantiated_expr,
                    lower_bound=instantiated_lb,
                    upper_bound=instantiated_ub,
                )
                _create_constraint(
                    self.solver,
                    self.context,
                    instantiated_constraint,
                )

    def _create_objectives(self) -> None:
        for component in self.context.network.all_components:
            model = component.model

            for objective in self.context.build_strategy.get_objectives(model):
                if objective is not None:
                    _create_objective(
                        self.solver,
                        self.context,
                        component,
                        self.context.risk_strategy(objective),
                    )

    def export_as_mps(self) -> str:
        return self.solver.ExportModelAsMpsFormat(fixed_format=True, obfuscated=False)

    def export_as_lp(self) -> str:
        return self.solver.ExportModelAsLpFormat(obfuscated=False)


def build_problem(
    network: Network,
    database: DataBase,
    block: TimeBlock,
    scenarios: int,
    *,
    problem_name: str = "optimization_problem",
    border_management: BlockBorderManagement = BlockBorderManagement.CYCLE,
    solver_id: str = "SCIP",
    build_strategy: ModelSelectionStrategy = MergedProblemStrategy(),
    risk_strategy: RiskManagementStrategy = UniformRisk(),
    decision_tree_node: str = "",
    use_full_var_name: bool = True,
) -> OptimizationProblem:
    """
    Entry point to build the optimization problem for a time period.
    """
    solver: lp.Solver = lp.Solver.CreateSolver(solver_id)

    database.requirements_consistency(network)

    opt_context = OptimizationContext(
        network,
        database,
        block,
        scenarios,
        border_management,
        build_strategy,
        risk_strategy,
        decision_tree_node,
        use_full_var_name,
    )

    return OptimizationProblem(problem_name, solver, opt_context)


def fusion_problems(
    masters: List[OptimizationProblem], coupler: OptimizationProblem
) -> OptimizationProblem:
    if len(masters) == 1:
        # Nothing to fusion. Just past down the master
        return masters[0]

    root_master = coupler
    root_master.name = "master"

    root_vars: Dict[str, lp.Variable] = dict()
    root_constraints: Dict[str, lp.Constraint] = dict()
    root_objective = root_master.solver.Objective()

    # We stock the coupler's variables to check for
    # same name variables in the masters
    for var in root_master.solver.variables():
        root_vars[var.name()] = var

    for master in masters:
        context = master.context
        objective = master.solver.Objective()

        for var in master.solver.variables():
            # If variable not already in coupler, we add it
            # Otherwise we update its upper and lower bounds
            if var.name() not in root_vars:
                root_var = root_master.solver.NumVar(var.lb(), var.ub(), var.name())
                root_master.context._solver_variables[var.name()] = SolverVariableInfo(
                    var.name(),
                    len(root_master.context._solver_variables),
                    context._solver_variables[var.name()].is_in_objective,
                )
            else:
                root_var = root_vars[var.name()]
                root_var.SetLb(var.lb())
                root_var.SetUb(var.ub())
                root_master.context._solver_variables[
                    var.name()
                ].is_in_objective = context._solver_variables[
                    var.name()
                ].is_in_objective

            for cstr in master.solver.constraints():
                coeff = cstr.GetCoefficient(var)
                # If variable present in constraint, we add the constraint to root
                if coeff != 0:
                    key = f"{master.name}_{cstr.name()}"
                    if key not in root_constraints:
                        root_constraints[key] = root_master.solver.Constraint(
                            cstr.Lb(), cstr.Ub(), key
                        )
                    root_cstr = root_constraints[key]
                    root_cstr.SetCoefficient(root_var, coeff)

            obj_coeff = objective.GetCoefficient(var)
            if obj_coeff != 0:
                root_objective.SetCoefficient(root_var, obj_coeff)

    return root_master
