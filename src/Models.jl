"""
Defines the abstract type DiscreteChoiceModel and the generic interface for models in DCM.jl.

This module establishes the base interface all discrete choice models must follow, including predict, loglikelihood, update_model, and estimate. Specific models (e.g., Logit, Mixed Logit) must subtype DiscreteChoiceModel and implement these methods.
"""
abstract type DiscreteChoiceModel end

function predict(model::DiscreteChoiceModel)
    error("predict not implemented for $(typeof(model))")
end

function loglikelihood(model::DiscreteChoiceModel, choices::Vector{Int})
    error("loglikelihood not implemented for $(typeof(model))")
end

function update_model(model::DiscreteChoiceModel, θ::Vector{Float64}, free_names, fixed_names, init_values)
    error("update_model not implemented for $(typeof(model))")
end

function estimate(model::DiscreteChoiceModel, choicevar; verbose = true)
    error("update_model not implemented for $(typeof(model))")
end

include("models/LogitModel.jl")
# include("MixedLogitModel.jl")

export LogitModel, estimate, predict, loglikelihood