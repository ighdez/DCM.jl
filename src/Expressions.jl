export Parameter, Variable, DCMExpression, logit_prob

abstract type DCMExpression end

struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool
end

struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}  # For panel/individual data
end

struct DCMSum <: DCMExpression
    left::DCMExpression
    right::DCMExpression
end

struct DCMMult <: DCMExpression
    left::DCMExpression
    right::DCMExpression
end

struct DCMExp <: DCMExpression
    arg::DCMExpression
end

import Base: +, *, exp

+(a::DCMExpression, b::DCMExpression) = DCMSum(a, b)
*(a::DCMExpression, b::DCMExpression) = DCMMult(a, b)
exp(a::DCMExpression) = DCMExp(a)

function Parameter(name::Symbol; value=0.0, fixed=false)
    return DCMParameter(name, value, fixed)
end

function Variable(name::Symbol; index=nothing)
    return DCMVariable(name, index)
end

function logit_prob(utilities::Vector{<:DCMExpression}, data::Dict{Symbol, Vector{Float64}}, params::Dict{Symbol, Float64})
    utils = [evaluate(U, data, params) for U in utilities]  # Vector of vectors
    exp_utils = [exp.(u) for u in utils]                    # Element-wise exp
    denom = reduce(+, exp_utils)                            # Vector: denominator for each observation
    return [eu ./ denom for eu in exp_utils]                # Vector of choice probability vectors
end