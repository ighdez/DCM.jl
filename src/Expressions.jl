"""
Defines symbolic expression types and evaluation methods for Discrete Choice Models.

This module implements the symbolic algebra system used to specify and evaluate utility functions in DCM.jl. Core types include symbolic representations for parameters, variables, sums, multiplications, exponentials, and comparisons.
"""

"""
abstract type DCMExpression
Base type for all symbolic expressions in the DCM system. All symbolic objects must inherit from this type.
"""
abstract type DCMExpression end
abstract type DCMBinary <: DCMExpression end
abstract type DCMUnary <: DCMExpression end

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

struct DCMMinus <: DCMUnary
    arg::DCMExpression
end

import Base: ==, +, *, exp, -


==(a::DCMExpression, b::Real) = DCMEqual(a, b)
+(a::DCMExpression, b::DCMExpression) = DCMSum(a, b)
*(a::DCMExpression, b::DCMExpression) = DCMMult(a, b)
exp(a::DCMExpression) = DCMExp(a)
-(a::DCMExpression) = DCMMinus(a)

"""
struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool

Represents a named parameter in a utility expression.

Fields:

* `name`: Symbol, name of the parameter
* `value`: Float64, initial value of the parameter
* `fixed`: Bool, whether the parameter is fixed during estimation
  """
struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool
end


"""
function Parameter(name::Symbol; value=0.0, fixed::Bool=false)

Convenient constructor for `DCMParameter`.

# Arguments

* `name`: Symbol name of the parameter
* `value`: initial value (default 0.0)
* `fixed`: whether the parameter is fixed (default false)

# Returns

A `DCMParameter` object.
"""
function Parameter(name::Symbol; value=0.0, fixed::Bool=false)
    return DCMParameter(name, value, fixed)
end

"""
struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}

Represents a data variable used in utility expressions.

Fields:

* `name`: Symbol, corresponding to a column in the dataset
* `index`: optional integer to reference individual data in panel settings
"""

struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}  # For panel/individual data
end


"""
function Variable(name::Symbol; index=nothing)

Convenient constructor for `DCMVariable`.

# Arguments

* `name`: Symbol name of the variable
* `index`: Optional individual index (default: `nothing`)

# Returns

A `DCMVariable` object.
"""
function Variable(name::Symbol; index=nothing)
    return DCMVariable(name, index)
end

"""
Represents a random draw term in a symbolic expression.
- `name`: Symbol used to identify the draw source (e.g., :time)
"""
struct DCMDraw <: DCMExpression
    name::Symbol
end

function Draw(name::Symbol)
    return DCMDraw(name)
end

"""
function evaluate(expr::DCMExpression, data::DataFrame, params::Dict{Symbol, <:Real})

Evaluates a symbolic utility expression for all observations in a DataFrame.

# Arguments

* `expr`: a symbolic expression of type `DCMExpression`
* `data`: a `DataFrame` with the data for all individuals/alternatives
* `params`: dictionary mapping parameter symbols to values

# Returns

A vector of numeric values representing the evaluated expression.
"""
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
    elseif expr isa DCMMinus
        return -evaluate(expr.arg, data, params)
    else
        error("Unknown expression type")
    end
end

"""
Extended evaluate function for symbolic expressions using draws.
# Arguments:
- expr: symbolic expression
- data: DataFrame with variables
- params: Dict of fixed parameter values (Symbol => Real)
- draws: Dict of draws (Symbol => Matrix NxR)

Returns a Matrix{Float64} of size NxR
"""
function evaluate(expr::DCMExpression, data::DataFrame, params::Dict{Symbol,<:Real}, draws::Dict{Symbol, Matrix{Float64}})
    N = nrow(data)
    R = first(values(draws)).size[2]

    if expr isa DCMParameter
        return fill(params[expr.name], N, R)
    elseif expr isa DCMVariable
        col = data[:, expr.name]
        return repeat(reshape(col, N, 1), 1, R)
    elseif expr isa DCMDraw
        return draws[expr.name]  # N × R
    elseif expr isa DCMSum
        return evaluate(expr.left, data, params, draws) .+ evaluate(expr.right, data, params, draws)
    elseif expr isa DCMMult
        return evaluate(expr.left, data, params, draws) .* evaluate(expr.right, data, params, draws)
    elseif expr isa DCMExp
        return exp.(evaluate(expr.arg, data, params, draws))
    elseif expr isa DCMEqual
        left_val = evaluate(expr.left, data, params, draws)
        return Float64.(left_val .== expr.right)
        elseif expr isa DCMMinus
    return -evaluate(expr.arg, data, params, draws)
    else
        error("Unknown expression type")
    end
end

"""
function logit_prob(utilities::Vector{<:DCMExpression}, data::DataFrame,
    params::Dict{Symbol, <:Real}, availability::Vector{<:AbstractVector{Bool}})

Computes choice probabilities for the Multinomial Logit model.

# Arguments

* `utilities`: vector of symbolic utility expressions (one per alternative)
* `data`: DataFrame of input data
* `params`: parameter dictionary with values for evaluation
* `availability`: vector of boolean vectors indicating available alternatives

# Returns

A vector of vectors, each inner vector representing choice probabilities for each alternative per observation.
"""
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

export Parameter, Variable, evaluate, logit_prob, Draw