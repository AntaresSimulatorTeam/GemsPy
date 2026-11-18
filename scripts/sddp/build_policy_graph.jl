# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

# Example: builds an SDDP.jl policy graph from a GemsPy parametric export
# (`gems_runner.parametric.export_parametric_problems`, parallel-subproblems
# mode). One stage per block; the noise of a stage is the Monte-Carlo scenario.
#
# Usage:
#   julia --project build_policy_graph.jl path/to/export
#
# Dependencies: SDDP, JuMP, HiGHS, JSON3, DataFrames, Parquet2.
#
# TODO: declare the state variables linking consecutive stages (e.g. the
# storage level at the end of block i and at the start of block i + 1) with
# `SDDP.State`, using the variable names of the MPS files. Without states, the
# stages are independent and SDDP gives the expected cost of each block.

using DataFrames
using HiGHS
using JSON3
using JuMP
using Parquet2
using SDDP

const export_dir = abspath(get(ARGS, 1, "."))
const manifest = JSON3.read(read(joinpath(export_dir, "manifest.json"), String))

manifest.format_version == 1 || error("Unsupported format_version.")

const blocks = collect(Int, manifest.blocks)
const scenario_ids = collect(Int, manifest.scenarios)
# Scenarios are equiprobable.
const probabilities = fill(1 / length(scenario_ids), length(scenario_ids))

block_file(b, name) = joinpath(export_dir, "block_$b", name)

isfile(block_file(first(blocks), "problem.mps")) ||
    error("This example handles frontal and parallel-subproblems exports only.")

"Entries of one problem, grouped by scenario."
function read_parameters(path::String)
    df = DataFrame(Parquet2.Dataset(path); copycols = false)
    return Dict(s => df[df.scenario.==s, :] for s in scenario_ids)
end

const parameters =
    [read_parameters(block_file(b, "problem_parameters.parquet")) for b in blocks]

"Copy the problem of `mps` into the subproblem `sp`; return its objective."
function copy_problem!(sp::JuMP.Model, mps::String)
    src = read_from_file(mps)
    index_map = MOI.copy_to(backend(sp), backend(src))
    objective = objective_function(src)
    # The objective goes through @stageobjective, not the copied model.
    set_objective_function(sp, zero(AffExpr))
    return AffExpr(
        objective.constant,
        Pair{VariableRef,Float64}[
            VariableRef(sp, index_map[index(v)]) => c for
            (c, v) in linear_terms(objective)
        ]...,
    )
end

"Apply the entries of one scenario; return the stage objective."
function apply!(sp::JuMP.Model, entries::DataFrame, nominal::AffExpr)
    objective = copy(nominal)
    for e in eachrow(entries)
        if e.kind == "rhs"
            set_normalized_rhs(constraint_by_name(sp, e.row), e.value)
        elseif e.kind == "coef"
            set_normalized_coefficient(
                constraint_by_name(sp, e.row),
                variable_by_name(sp, e.col),
                e.value,
            )
        elseif e.kind == "lower"
            set_lower_bound(variable_by_name(sp, e.col), e.value)
        elseif e.kind == "upper"
            set_upper_bound(variable_by_name(sp, e.col), e.value)
        elseif e.kind == "obj"
            objective.terms[variable_by_name(sp, e.col)] = e.value
        elseif e.kind == "obj_const"
            objective.constant = e.value
        end
    end
    return objective
end

model = SDDP.LinearPolicyGraph(;
    stages = length(blocks),
    sense = :Min,
    lower_bound = 0.0,  # adapt: a lower bound of the cost of the future stages
    optimizer = HiGHS.Optimizer,
) do sp, stage
    nominal = copy_problem!(sp, block_file(blocks[stage], "problem.mps"))
    SDDP.parameterize(sp, scenario_ids, probabilities) do s
        @stageobjective(sp, apply!(sp, parameters[stage][s], nominal))
    end
end

SDDP.train(model; iteration_limit = 10)
println("Lower bound: ", SDDP.calculate_bound(model))
