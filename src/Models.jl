using DataFrames

abstract type DiscreteChoiceModel end

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