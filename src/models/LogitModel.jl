using DataFrames

struct LogitModel <: DiscreteChoiceModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict
    availability::Vector{<:AbstractVector{Bool}}
end

function LogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    parameters::Dict = Dict(),
    availability::Vector{<:AbstractVector{Bool}} = []
)
    return LogitModel(utilities, data, parameters, availability)
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

function update_model(model::LogitModel, θ, free_names, fixed_names, init_values)
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