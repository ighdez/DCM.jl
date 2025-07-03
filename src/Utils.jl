"""
Extracts all distinct parameters from a list of utility expressions.

Traverses the expression trees and returns all unique instances of `DCMParameter`.

# Arguments
- `utilities::Vector{<:DCMExpression}`: vector of symbolic utility expressions

# Returns
- `Vector{DCMParameter}`: unique parameters used in the utilities
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

"""
Extracts all variable names used in a list of utility expressions.

Traverses the expression trees to find all `DCMVariable` symbols.

# Arguments
- `utilities::Vector{<:DCMExpression}`: vector of symbolic utility expressions

# Returns
- `Vector{Symbol}`: names of variables appearing in the expressions
"""
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
Recursively collects all unique draw names (`Symbol`) from a symbolic expression.

Used for identifying the random terms in Mixed Logit specifications.

# Arguments
- `expr::DCMExpression`: symbolic utility expression

# Returns
- `Vector{Symbol}`: names of draws used in the expression
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
Pretty-prints estimation results including estimates, standard errors, t-stats, and p-values.

This function is designed for displaying results from a Logit or Mixed Logit model
estimated via the `estimate` function.

# Arguments
- `results::NamedTuple`: named tuple returned from model estimation, containing fields:
    - `parameters`: Dict of estimated parameter values
    - `std_errors`: Dict of classical standard errors
    - `rob_std_errors`: Dict of robust standard errors (optional)
    - `loglikelihood`: log-likelihood value
    - `iters`: number of iterations
    - `converged`: convergence flag
    - `estimation_time`: runtime in seconds

# Returns
- Nothing. Prints output to console.
"""
function summarize_results(results::NamedTuple)
    params = results.parameters
    se_dict = results.std_errors
    se_robust = results.rob_std_errors
    ll = results.loglikelihood
    iters = results.iters
    converged = results.converged
    estimation_time = results.estimation_time

    println("Estimation Results\n==================\n")

    println("Classic Standard Errors")
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

    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))

    for (name, value) in sort(collect(params); by=first)
        if haskey(se_robust, name)
            se = se_robust[name]
            t = value / se
            p = 2 * (1 - cdf(Normal(), abs(t)))
            println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
        else
            println(@sprintf("%-20s %10.4f %12s %10s %10s", string(name), value, "NA", "NA", "NA"))
        end
    end

    println("\nModel Summary")
    println(@sprintf("Log-likelihood at optimum  : %10.4f", ll))
    println(@sprintf("Iterations                 : %10s", iters))
    println(@sprintf("Converged                  : %10s", converged))
    println(@sprintf("Estimation time (seconds)  : %10.2f", estimation_time))
end

"""
Pretty-prints results from evaluating derived expressions (e.g., WTP, elasticities).

Used to display values and their standard errors (e.g., computed via Delta method).

# Arguments
- `results::Dict{Symbol,<:NamedTuple}`: dictionary with results per expression, where each value has fields `value` and `std_error`

# Returns
- Nothing. Prints output to console.
"""
function summarize_expressions(results::Dict{Symbol,<:NamedTuple})
    println("Expression Evaluation\n======================\n")

    println("Classic Standard Errors")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))

    for (name, output) in sort(collect(results); by=first)
        value = output.value
        se = output.std_error
        t = value / se
        p = 2 * (1 - cdf(Normal(), abs(t)))
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
    end

    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))

    for (name, output) in sort(collect(results); by=first)
        value = output.value
        se = output.robust_std_error
        t = value / se
        p = 2 * (1 - cdf(Normal(), abs(t)))
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
    end
end

export summarize_results, summarize_expressions