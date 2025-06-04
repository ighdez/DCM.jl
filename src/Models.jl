export LogitModel

struct LogitModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict{Symbol, <:Real}
    availability::Vector{<:AbstractVector{Bool}}
    
    LogitModel(utilities::Vector{<:DCMExpression}; 
                    data::DataFrame,
                    availability::Vector{<:AbstractVector{Bool}},
                    parameters::Dict{Symbol, <:Real}) = new(utilities, data, parameters, availability)
end