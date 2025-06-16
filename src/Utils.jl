"""
Helper functions for working with utility expressions and model results.

This module defines support tools such as parameter collection from symbolic utilities and result summarization.
"""

"""
function collect_parameters(utilities::Vector{<:DCMExpression})

Extracts all distinct parameters from a vector of symbolic utility expressions.
Traverses each expression and returns a list of unique `DCMParameter` instances.

# Arguments

* `utilities`: vector of utility expressions (`Vector{<:DCMExpression}`)

# Returns

Vector of `DCMParameter` instances
"""
function collect_parameters(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, DCMParameter}()
    function visit(expr)
        if expr isa DCMParameter
            seen[expr.name] = expr
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(values(seen))
end

function collect_variables(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, Bool}()
    function visit(expr)
        if expr isa DCMVariable
            seen[expr.name] = true
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(keys(seen))
end


"""
    collect_draws(expr::DCMExpression) -> Vector{Symbol}

Recursively collects all unique draw names (`Symbol`) from a utility expression tree.

This function traverses any expression involving `DCMDraw` nodes (i.e., placeholders
for random draws) and returns a vector of the names of these draws, with duplicates removed.

This is typically used when building a `MixedLogitModel`, to determine which draws
need to be generated for estimation or simulation.

# Arguments
- `expr`: A `DCMExpression`, such as a utility function involving random parameters.

# Returns
- A `Vector{Symbol}` with the names of all unique `DCMDraw` objects found in the expression.
"""
function collect_draws(utilities::Vector{<:DCMExpression})
    seen = Dict{Symbol, Bool}()
    function visit(expr)
        if expr isa DCMDraw
            seen[expr.name] = true
        elseif expr isa DCMBinary
            visit(expr.left)
            visit(expr.right)
        elseif expr isa DCMUnary
            visit(expr.arg)
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(keys(seen))
end

"""
function summarize_results(results)

Pretty-prints estimation results including estimates, standard errors, t-stats, and p-values.

# Arguments

* `results`: named tuple returned from an estimation (should include keys `parameters`, `std_errors`, `loglikelihood`, `iters`, `converged`, and `estimation_time`)

# Returns

Nothing. Prints formatted output to stdout.
"""

function summarize_results(results)
    params = results.parameters
    se_dict = results.std_errors
    ll = results.loglikelihood
    iters = results.iters
    converged = results.converged
    estimation_time = results.estimation_time

    println("Estimation Results\n==================\n")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))

    for (name, value) in sort(collect(params); by=first)
        if haskey(se_dict, name)
            se = se_dict[name]
            t = value / se
            p = 2 * (1 - cdf(Normal(), abs(t)))
            println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
        else
            println(@sprintf("%-20s %10.4f %12s %10s %10s", string(name), value, "NA", "NA", "NA"))
        end
    end

    println(@sprintf("Log-likelihood at optimum  : %10.4f", ll))
    println(@sprintf("Iterations                 : %10s", iters))
    println(@sprintf("Converged                  : %10s", converged))
    println(@sprintf("Estimation time (seconds)  : %10.2f", estimation_time))

end

export summarize_results, collect_parameters