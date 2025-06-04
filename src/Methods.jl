export evaluate, predict, loglikelihood, estimate

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

function estimate(model::LogitModel, choices::Vector{Int})
    param_names = collect(keys(model.parameters))
    θ0 = [model.parameters[p] for p in param_names]

    function objective(θ)
        param_map = Dict(param_names .=> θ)
        updated_model = LogitModel(model.utilities; data=model.data, parameters=param_map, availability=model.availability)
        return -loglikelihood(updated_model, choices)
    end

    result = Optim.optimize(objective, θ0, Optim.BFGS(); autodiff = :forward)

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict(param_names .=> θ̂)

    return (
        result = result,
        parameters = estimated_params,
        loglikelihood = -Optim.minimum(result)
    )
end
