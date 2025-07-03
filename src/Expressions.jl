### Expressions.jl — Symbolic Expressions for Discrete Choice Models

"""
Abstract type for all symbolic expressions in the DCM system.
All symbolic objects (parameters, variables, operators) must inherit from this type.
"""
abstract type DCMExpression end

abstract type DCMBinary <: DCMExpression end
abstract type DCMUnary <: DCMExpression end

"""
Represents a constant literal value in an expression tree.

# Fields
- `value::Float64`: the constant numeric value
"""
struct DCMLiteral <: DCMExpression
    value::Float64
end

"""
Represents a symbolic equality comparison between an expression and a numeric value.

# Fields
- `left::DCMExpression`: left-hand side symbolic expression
- `right::Real`: right-hand side numeric value
"""
struct DCMEqual <: DCMBinary
    left::DCMExpression
    right::Real
end

"""
Symbolic addition of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMSum <: DCMBinary
    left::DCMExpression
    right::DCMExpression
end

"""
Symbolic multiplication of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMMult <: DCMBinary
    left::DCMExpression
    right::DCMExpression
end

"""
Symbolic division of two expressions.

# Fields
- `left`, `right`: symbolic expressions
"""
struct DCMDiv <: DCMBinary
    left::DCMExpression
    right::DCMExpression
end

"""
Symbolic exponential of an expression.

# Fields
- `arg`: symbolic expression
"""
struct DCMExp <: DCMUnary
    arg::DCMExpression
end

"""
Symbolic logarithm of an expression.

# Fields
- `arg`: symbolic expression
"""
struct DCMLog <: DCMUnary
    arg::DCMExpression
end

"""
Symbolic negation of an expression (unary minus).

# Fields
- `arg`: symbolic expression
"""
struct DCMMinus <: DCMUnary
    arg::DCMExpression
end

# Operator overloads

import Base: ==, +, *, /, exp, log, -
==(a::DCMExpression, b::Real) = DCMEqual(a, b)
+(a::DCMExpression, b::DCMExpression) = DCMSum(a, b)
*(a::DCMExpression, b::DCMExpression) = DCMMult(a, b)
/(a::DCMExpression, b::DCMExpression) = DCMDiv(a, b)
exp(a::DCMExpression) = DCMExp(a)
log(a::DCMExpression) = DCMLog(a)
-(a::DCMExpression) = DCMMinus(a)

"""
Represents a named parameter in a utility expression.

# Fields
- `name::Symbol`: parameter name
- `value::Float64`: initial value
- `fixed::Bool`: whether the parameter is fixed during estimation
"""
struct DCMParameter <: DCMExpression
    name::Symbol
    value::Float64
    fixed::Bool
end

"""
Constructor for `DCMParameter`.

# Arguments
- `name::Symbol`: parameter name
- `value=0.0`: initial value (default: 0.0)
- `fixed::Bool=false`: fixed during estimation? (default: false)
"""
function Parameter(name::Symbol; value=0.0, fixed::Bool=false)
    return DCMParameter(name, value, fixed)
end

"""
Represents a data variable used in utility expressions.

# Fields
- `name::Symbol`: name of the variable, typically a column in the dataset
- `index::Union{Nothing, Int}`: optional index for panel data structure
"""
struct DCMVariable <: DCMExpression
    name::Symbol
    index::Union{Nothing, Int}  # For panel/individual data
end

"""
Constructor for `DCMVariable`.

# Arguments
- `name::Symbol`: variable name (must match column name in data)
- `index=nothing`: optional index for individual-specific variables

# Returns
- `DCMVariable` object
"""
function Variable(name::Symbol; index=nothing)
    return DCMVariable(name, index)
end


"""
Represents a symbolic placeholder for random draws in Mixed Logit models.

# Fields
- `name::Symbol`: name of the draw (e.g., `:draw_normal_time`)
"""
struct DCMDraw <: DCMExpression
    name::Symbol
end

"""
Constructor for `DCMDraw`.

# Arguments
- `name::Symbol`: name of the random draw

# Returns
- `DCMDraw` object
"""
function Draw(name::Symbol)
    return DCMDraw(name)
end

"""
Evaluates a symbolic utility expression for all observations in a dataset.

# Arguments
- `expr::DCMExpression`: symbolic expression to evaluate
- `data::DataFrame`: dataset with values for variables
- `params::AbstractDict`: dictionary with parameter names and values

# Returns
- `Vector{Float64}`: evaluated numeric result for each observation
"""
function evaluate(expr::DCMExpression, data::DataFrame, params::AbstractDict)
    if expr isa DCMParameter
        return fill(params[expr.name], nrow(data))
    elseif expr isa DCMVariable
        return data[:, expr.name]
    elseif expr isa DCMSum
        return evaluate(expr.left, data, params) .+ evaluate(expr.right, data, params)
    elseif expr isa DCMMult
        return evaluate(expr.left, data, params) .* evaluate(expr.right, data, params)
    elseif expr isa DCMDiv
        return evaluate(expr.left, data, params) ./ evaluate(expr.right, data, params)
    elseif expr isa DCMExp
        return exp.(evaluate(expr.arg, data, params))
    elseif expr isa DCMLog
        return log.(evaluate(expr.arg, data, params))
    elseif expr isa DCMEqual
        left_val = evaluate(expr.left, data, params)
        return ifelse.(left_val .== expr.right, one(eltype(left_val)), zero(eltype(left_val)))
    elseif expr isa DCMMinus
        return -evaluate(expr.arg, data, params)
    elseif expr isa DCMLiteral
        return fill(expr.value, nrow(data))
    else
        error("Unknown expression type")
    end
end

"""
Evaluates a symbolic expression for all observations and draws in Mixed Logit models.

This version supports replication of data over simulation draws and handles parameter values,
random draws, and data variables.

# Arguments
- `expr::DCMExpression`: the symbolic expression to evaluate
- `data::DataFrame`: the dataset (N rows)
- `params::AbstractDict`: mapping from parameter names to values
- `draws::AbstractDict`: dictionary of random draws, each as an `N × R` array

# Returns
- `Array{Float64, 2}`: evaluated result of shape `N × R`
"""
function evaluate(expr::DCMExpression, data::DataFrame, params::AbstractDict, draws::AbstractDict)
    result = begin
        if expr isa DCMParameter
            N, R = size(first(values(draws)))
            fill(params[expr.name], N, R)

        elseif expr isa DCMVariable
            _, R = size(first(values(draws)))
            repeat(data[:, expr.name], 1, R)

        elseif expr isa DCMDraw
            draws[expr.name]

        elseif expr isa DCMSum
            left = evaluate(expr.left, data, params, draws)
            right = evaluate(expr.right, data, params, draws)
            T = promote_type(eltype(left), eltype(right))
            out = Array{T}(undef, size(left))
            @. out = left + right
            out

        elseif expr isa DCMMult
            left = evaluate(expr.left, data, params, draws)
            right = evaluate(expr.right, data, params, draws)
            T = promote_type(eltype(left), eltype(right))
            out = Array{T}(undef, size(left))
            @. out = left * right
            out

        elseif expr isa DCMDiv
            left = evaluate(expr.left, data, params, draws)
            right = evaluate(expr.right, data, params, draws)
            T = promote_type(eltype(left), eltype(right))
            out = Array{T}(undef, size(left))
            @. out = left / right
            out

        elseif expr isa DCMExp
            arg = evaluate(expr.arg, data, params, draws)
            T = eltype(arg)
            out = Array{T}(undef, size(arg))
            @. out = exp(arg)
            out

        elseif expr isa DCMLog
            arg = evaluate(expr.arg, data, params, draws)
            T = eltype(arg)
            out = Array{T}(undef, size(arg))
            @. out = log(arg)
            out

        elseif expr isa DCMEqual
            left_val = evaluate(expr.left, data, params, draws)
            T = eltype(left_val)
            out = Array{T}(undef, size(left_val))
            @. out = ifelse(left_val == expr.right, one(T), zero(T))
            out

        elseif expr isa DCMMinus
            arg = evaluate(expr.arg, data, params, draws)
            T = eltype(arg)
            out = Array{T}(undef, size(arg))
            @. out = -arg
            out

        elseif expr isa DCMLiteral
            N, R = size(first(values(draws)))
            fill(expr.value, N, R)

        else
            error("Unknown expression type")
        end
    end

    return result
end

export Parameter, Variable, Draw, evaluate