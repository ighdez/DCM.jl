function evaluate(expr::DCMExpression, data::DataFrame, params::Dict{Symbol, <:Real})
    if expr isa DCMParameter
        return fill(params[expr.name], nrow(data))
    elseif expr isa DCMVariable
        return data[:, expr.name]
    elseif expr isa DCMSum
        return evaluate(expr.left, data, params) .+ evaluate(expr.right, data, params)
    elseif expr isa DCMMult
        return evaluate(expr.left, data, params) .* evaluate(expr.right, data, params)
    elseif expr isa DCMExp
        return exp.(evaluate(expr.arg, data, params))
    elseif expr isa DCMEqual
        left_val = evaluate(expr.left, data, params)
        return Float64.(left_val .== expr.right)
    else
        error("Unknown expression type")
    end
end

function predict(model::LogitModel)
    return logit_prob(model.utilities, model.data, model.parameters, model.availability)
end

function loglikelihood(model::LogitModel, choices::Vector{Int})
    probs = predict(model)  # list of vectors: one per alternative
    loglik = 0.0
    N = length(choices)
    for n in 1:N
        chosen_alt = choices[n]
        chosen_prob = probs[chosen_alt][n]
        loglik += log(chosen_prob)
    end
    return loglik
end

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

function estimate(model::LogitModel, choicevar; verbose = true)

    if any(ismissing, choicevar)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choicevar)
    
    params = collect_parameters(model.utilities)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    # Get free/fixed masks
    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    # Initial guess only for free params
    θ0 = [init_values[n] for n in free_names]

    function update_model(θ)
        full_values = Dict{Symbol, Real}()
        for (i, name) in enumerate(free_names)
            full_values[name] = θ[i]
        end
        for name in fixed_names
            full_values[name] = init_values[name]
        end
        return LogitModel(model.utilities;
            data=model.data,
            parameters=full_values,
            availability=model.availability)
    end

    function objective(θ)
        updated = update_model(θ)
        return -loglikelihood(updated, choices)
    end

    if verbose
        println("Starting optimization routine...")
    end

    t_start = time()
    result = Optim.optimize(objective, θ0, Optim.BFGS(),Optim.Options(show_trace = verbose, iterations = 1000); autodiff = :forward)

    if verbose & Optim.converged(result)
        println("Converged")
    end

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict{Symbol, Real}()

    for (i, name) in enumerate(free_names)
        estimated_params[name] = θ̂[i]
    end

    for name in fixed_names
        estimated_params[name] = init_values[name]
    end

    # Hessian
    if verbose
        println("Computing Standard Errors")
    end

    H = ForwardDiff.hessian(objective, θ̂)
    vcov = inv(H)
    std_errors = sqrt.(diag(vcov))
    t_end = time()

    se = Dict{Symbol, Real}()

    for (i, name) in enumerate(free_names)
        se[name] = std_errors[i]
    end

    return (
        result = result,
        parameters = estimated_params,
        std_errors = se,
        vcov = vcov,
        loglikelihood = -Optim.minimum(result),
        iters = Optim.iterations(result),
        converged = Optim.converged(result),
        estimation_time = t_end - t_start
    )
end

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