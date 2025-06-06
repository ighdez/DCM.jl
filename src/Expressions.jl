abstract type DCMExpression end
abstract type DCMBinary <: DCMExpression end
abstract type DCMUnary <: DCMExpression end

struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool
end

struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}  # For panel/individual data
end

function Parameter(name::Symbol; value=0.0, fixed::Bool=false)
    return DCMParameter(name, value, fixed)
end

function Variable(name::Symbol; index=nothing)
    return DCMVariable(name, index)
end

struct DCMEqual <: DCMBinary
    left::DCMExpression
    right::Real
end

struct DCMSum <: DCMBinary
    left::DCMExpression
    right::DCMExpression
end

struct DCMMult <: DCMBinary
    left::DCMExpression
    right::DCMExpression
end

struct DCMExp <: DCMUnary
    arg::DCMExpression
end

import Base: ==, +, *, exp

==(a::DCMExpression, b::Real) = DCMEqual(a, b)
+(a::DCMExpression, b::DCMExpression) = DCMSum(a, b)
*(a::DCMExpression, b::DCMExpression) = DCMMult(a, b)
exp(a::DCMExpression) = DCMExp(a)

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

function logit_prob(utilities::Vector{<:DCMExpression}, data::DataFrame,
    params::Dict{Symbol, <:Real}, availability::Vector{<:AbstractVector{Bool}})
    Nalts = length(utilities)

    utils = [evaluate(U, data, params) for U in utilities]  # Vector of vectors
    
    exp_utils = [exp.(u) for u in utils]                    # Element-wise exp
    for j in 1:Nalts
        exp_utils[j] = [avail ? exp(u) : 0.0 for (u, avail) in zip(utils[j], availability[j])]
    end

    denom = reduce(+, exp_utils)                            # Vector: denominator for each observation
    return [eu ./ denom for eu in exp_utils]                # Vector of choice probability vectors
end

export Parameter, Variable