"""
Defines the abstract type DiscreteChoiceModel and the generic interface for models in DCM.jl.

This module establishes the base interface all discrete choice models must follow, including predict, loglikelihood, and estimate. Specific models (e.g., Logit, Mixed Logit) must subtype DiscreteChoiceModel and implement these methods.
"""
abstract type DiscreteChoiceModel end

# function predict(model::DiscreteChoiceModel)
#     error("predict not implemented for $(typeof(model))")
# end

# function loglikelihood(model::DiscreteChoiceModel, choices)
#     error("loglikelihood not implemented for $(typeof(model))")
# end

# function estimate(model::DiscreteChoiceModel, choicevar; verbose = true)
#     error("estimate not implemented for $(typeof(model))")
# end

include("models/LogitModel.jl")
include("models/MixedLogit.jl")

export LogitModel, MixedLogitModel, estimate, predict, loglikelihood