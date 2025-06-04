export LogitModel

struct LogitModel
    utilities::Vector{<:DCMExpression}
    data::Dict{Symbol, Vector{Float64}}
    parameters::Dict{Symbol, Float64}
    availability::Vector{<:AbstractVector{Bool}}
    
    LogitModel(utilities::Vector{<:DCMExpression}; 
                    data::Dict{Symbol, Vector{Float64}},
                    availability::Vector{<:AbstractVector{Bool}},
                    parameters::Dict{Symbol, Float64}) = new(utilities, data, parameters, availability)
end