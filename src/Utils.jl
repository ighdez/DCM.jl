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
        elseif expr isa LogitModel
            for u in expr.utilities
                visit(u)
            end
        end
    end
    for u in utilities
        visit(u)
    end
    return collect(values(seen))
end

function collect_parameters(expr::DCMExpression)
    collect_parameters([expr])
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
Pretty-prints estimation results and optionally writes them to an Excel file.

Displays parameter estimates with classical and robust (sandwich) standard errors,
t-statistics and p-values. Optionally writes results to an Excel file (.xlsx) with
two sheets:
- "Estimates": full table of parameter results
- "Summary"  : log-likelihood, iterations, convergence status, and runtime.

# Arguments
- `results::NamedTuple`: Named tuple returned from model estimation, containing fields:
    - `parameters`: Dict of estimated parameter values
    - `std_errors`: Dict of classical standard errors
    - `rob_std_errors`: Dict of robust standard errors (optional)
    - `loglikelihood`: log-likelihood value
    - `iters`: number of iterations
    - `converged`: convergence flag
    - `estimation_time`: runtime in seconds
- `file::Union{String, Nothing}`: Optional path to export results as Excel file.

# Returns
- Nothing. Prints output to console and optionally writes Excel file.
"""
function summarize_results(results::NamedTuple; file::Union{String, Nothing}=nothing)
    params = results.parameters
    se_dict = results.std_errors
    se_robust = results.rob_std_errors
    ll = results.loglikelihood
    iters = results.iters
    converged = results.converged
    estimation_time = results.estimation_time

    println("Estimation Results\n==================\n")

    # Prepara DataFrame con resultados
    df = DataFrame(Parameter=String[], Estimate=Float64[],
                   StdError=Float64[], tStat=Float64[], PValue=Float64[],
                   RobustSE=Float64[], Robust_tStat=Float64[], Robust_PValue=Float64[])

    for (name, value) in sort(collect(params); by=first)
        # Clásicos
        se_c = get(se_dict, name, NaN)
        t_c  = value / se_c
        p_c  = 2 * (1 - cdf(Normal(), abs(t_c)))

        # Robustos
        se_r = get(se_robust, name, NaN)
        t_r  = value / se_r
        p_r  = 2 * (1 - cdf(Normal(), abs(t_r)))

        push!(df, (string(name), value, se_c, t_c, p_c, se_r, t_r, p_r))
    end

    # Imprime resultados clásicos
    println("Classic Standard Errors")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Parameter, row.Estimate, row.StdError, row.tStat, row.PValue))
    end

    # Imprime resultados robustos
    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Parameter", "Estimate", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Parameter, row.Estimate, row.RobustSE, row.Robust_tStat, row.Robust_PValue))
    end

    # Parámetros adicionales
    free_params = length(se_dict)
    N = results.N               # <- asegúrate de incluir esto en el NamedTuple
    ll0 = results.null_loglikelihood

    aic = -2 * ll + 2 * free_params
    bic = -2 * ll + log(N) * free_params
    rho2 = isfinite(ll0) ? 1 - ll / ll0 : NaN


    # Imprime resumen
    println("\nModel Summary")
    println(@sprintf("Log-likelihood at optimum  : %10.4f", ll))
    println(@sprintf("Null Log-likelihood.       : %10.4f", ll0))
    println(@sprintf("Iterations                 : %10d", iters))
    println(@sprintf("Converged                  : %10s", converged))
    println(@sprintf("Estimation time (seconds)  : %10.2f", estimation_time))
    println(@sprintf("Number of free parameters  : %10d", free_params))
    println(@sprintf("Number of observations     : %10d", N))
    println(@sprintf("AIC                        : %10.2f", aic))
    println(@sprintf("BIC                        : %10.2f", bic))
    println(@sprintf("Rho-squared (McFadden)     : %10.4f", rho2))

    # Exporta a Excel si se especifica
    if !isnothing(file)
        XLSX.openxlsx(file, mode="w") do xf
            # Hoja de resultados
            sheet = xf[1]
            XLSX.rename!(sheet,"Estimates")
            sheet1 = xf["Estimates"]

            # Escribir DataFrame celda por celda
            for (j, name) in enumerate(names(df))
                sheet1[1, j] = String(name)
            end
            for (i, row) in enumerate(eachrow(df))
                for (j, name) in enumerate(names(df))
                    sheet1[i+1, j] = row[name]
                end
            end

            # Hoja resumen
            XLSX.addsheet!(xf, "Summary")
            sheet2 = xf["Summary"]
            sheet2[1,1:2] = ["Log-likelihood at optimum", ll]
            sheet2[2,1:2] = ["Null Log-likelihood", ll0]
            sheet2[3,1:2] = ["Iterations", iters]
            sheet2[4,1:2] = ["Converged", converged]
            sheet2[5,1:2] = ["Estimation time (s)", estimation_time]
            sheet2[6,1:2] = ["Number of free parameters", free_params]
            sheet2[7,1:2] = ["Number of observations", N]
            sheet2[8,1:2] = ["AIC", aic]
            sheet2[9,1:2] = ["BIC", bic]
            sheet2[10,1:2] = ["Rho-squared", rho2]
        end
    end
end

"""
Pretty-prints results from evaluating derived expressions (e.g., WTP, elasticities),
and optionally writes them to an Excel file.

Displays values and their classical and robust standard errors, t-statistics and p-values.

# Arguments
- `results::Dict{Symbol,<:NamedTuple}`: dictionary with results per expression, where each value has fields `value`, `std_error`, and `robust_std_error`
- `file::Union{String, Nothing}`: Optional path to export results as Excel file.

# Returns
- Nothing. Prints output to console and optionally writes Excel file.
"""
function summarize_expressions(results::Dict{Symbol,<:NamedTuple}; file::Union{String, Nothing}=nothing)
    println("Expression Evaluation\n======================\n")

    df = DataFrame(Expression=String[], Value=Float64[],
                   StdError=Float64[], tStat=Float64[], PValue=Float64[],
                   RobustSE=Float64[], Robust_tStat=Float64[], Robust_PValue=Float64[])

    for (name, output) in sort(collect(results); by=first)
        value = output.value

        # Clásico
        se_c = output.std_error
        t_c  = value / se_c
        p_c  = 2 * (1 - cdf(Normal(), abs(t_c)))

        # Robusto
        se_r = output.robust_std_error
        t_r  = value / se_r
        p_r  = 2 * (1 - cdf(Normal(), abs(t_r)))

        push!(df, (string(name), value, se_c, t_c, p_c, se_r, t_r, p_r))
    end

    # Imprime resultados clásicos
    println("Classic Standard Errors")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Std. Error", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Expression, row.Value, row.StdError, row.tStat, row.PValue))
    end

    # Imprime resultados robustos
    println("\nRobust Standard Errors (Sandwich)")
    println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Robust SE", "t-Stat", "P-value"))
    println(repeat("-", 70))
    for row in eachrow(df)
        println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f",
            row.Expression, row.Value, row.RobustSE, row.Robust_tStat, row.Robust_PValue))
    end

    # Exportar a Excel si se especifica
    if !isnothing(file)
        XLSX.openxlsx(file, mode="w") do xf
            sheet = xf[1]
            XLSX.rename!(sheet,"Expressions")
            sheet = xf["Expressions"]

            # Escribir encabezados
            for (j, name) in enumerate(names(df))
                sheet[1, j] = String(name)
            end

            # Escribir filas
            for (i, row) in enumerate(eachrow(df))
                for (j, name) in enumerate(names(df))
                    sheet[i+1, j] = row[name]
                end
            end
        end
    end
end

# """
# Pretty-prints results from evaluating derived expressions (e.g., WTP, elasticities).

# Used to display values and their standard errors (e.g., computed via Delta method).

# # Arguments
# - `results::Dict{Symbol,<:NamedTuple}`: dictionary with results per expression, where each value has fields `value` and `std_error`

# # Returns
# - Nothing. Prints output to console.
# """
# function summarize_expressions(results::Dict{Symbol,<:NamedTuple})
#     println("Expression Evaluation\n======================\n")

#     println("Classic Standard Errors")
#     println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Std. Error", "t-Stat", "P-value"))
#     println(repeat("-", 70))

#     for (name, output) in sort(collect(results); by=first)
#         value = output.value
#         se = output.std_error
#         t = value / se
#         p = 2 * (1 - cdf(Normal(), abs(t)))
#         println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
#     end

#     println("\nRobust Standard Errors (Sandwich)")
#     println(@sprintf("%-20s %10s %14s %10s %10s", "Expression", "Value", "Robust SE", "t-Stat", "P-value"))
#     println(repeat("-", 70))

#     for (name, output) in sort(collect(results); by=first)
#         value = output.value
#         se = output.robust_std_error
#         t = value / se
#         p = 2 * (1 - cdf(Normal(), abs(t)))
#         println(@sprintf("%-20s %10.4f %12.4f %10.4f %10.4f", string(name), value, se, t, p))
#     end
# end



export summarize_results, summarize_expressions