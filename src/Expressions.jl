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
        return ifelse.(left_val .== expr.right, one(eltype(left_val)), zero(eltype(left_val)))
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
function evaluate(expr::DCMExpression, data::DataFrame, params::AbstractDict, draws::AbstractDict)
    N = nrow(data)
    R = size(first(values(draws)), 2)

    if expr isa DCMParameter
        return fill(params[expr.name], N, R)
    elseif expr isa DCMVariable
        col = data[:, expr.name]
        return repeat(reshape(col, N, 1), 1, R)
    elseif expr isa DCMDraw
        return draws[expr.name]
    elseif expr isa DCMSum
        return evaluate(expr.left, data, params, draws) .+ evaluate(expr.right, data, params, draws)
    elseif expr isa DCMMult
        return evaluate(expr.left, data, params, draws) .* evaluate(expr.right, data, params, draws)
    elseif expr isa DCMExp
        return exp.(evaluate(expr.arg, data, params, draws))
    elseif expr isa DCMEqual
        left_val = evaluate(expr.left, data, params, draws)
        return ifelse.(left_val .== expr.right, one(eltype(left_val)), zero(eltype(left_val)))
    elseif expr isa DCMMinus
        return -evaluate(expr.arg, data, params, draws)
    else
        error("Unknown expression type")
    end
end

export Parameter, Variable, evaluate, Draw