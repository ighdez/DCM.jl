# Expressions
export Variable, Coefficient, Expression, @var, @coef, +, -, *, /, build_expression

# Variable: refers to a column in the data
"""
Refers to a column in the data
"""
struct Variable
    name::Symbol
end

# Coefficient: symbolic parameter with optional bounds or initial value
"""
Symbolic parameter with optional bounds or initial value
"""
mutable struct Coefficient
    name::Symbol
    value::Float64
#    lower::Union{Nothing, Float64}
#    upper::Union{Nothing, Float64}
end

# Convenience constructors
macro var(name)
    esc(:($(Variable)(Symbol($name))))
end

macro coef(name, init=0.0)
    return :(Coefficient($(QuoteNode(name)), $init))
    # return :(Coefficient($(QuoteNode(name)), nothing, nothing))
end

# Abstract type for expressions
"""
Abstract type for expressions
"""
abstract type Expression end


# Expression leaf
"""
Expression leaf
"""
struct ExprLeaf <: Expression
    value::Union{Variable, Coefficient, Float64}
end

# Expression operators
"""
Expression operators
"""
struct ExprOp <: Expression
    op::Symbol
    left::Expression
    right::Expression
end

# Promote to expression leaf
"""
Promote to expression leaf
"""
promote_expr(x::Expression) = x
promote_expr(x::Union{Variable, Coefficient, Float64}) = ExprLeaf(x)

# Overload arithmetic
import Base: +, -, *, /

+(a, b) = ExprOp(:+, promote_expr(a), promote_expr(b))
-(a, b) = ExprOp(:-, promote_expr(a), promote_expr(b))
*(a, b) = ExprOp(:*, promote_expr(a), promote_expr(b))
/(a, b) = ExprOp(:/, promote_expr(a), promote_expr(b))

# Utility function
struct UtilityFunction
    name::Symbol
    expr::Expression
end