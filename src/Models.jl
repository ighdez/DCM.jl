export LogitModel

struct LogitModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict{Symbol, Float64}
    availability::Vector{<:AbstractVector{Bool}}
    
    LogitModel(utilities::Vector{<:DCMExpression}; 
                    data::DataFrame,
                    availability::Vector{<:AbstractVector{Bool}},
                    parameters::Dict{Symbol, Float64}) = new(utilities, data, parameters, availability)
end