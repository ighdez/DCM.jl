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

export LogitModel, estimate, predict