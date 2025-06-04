# import Expressions
export evaluate

"""
Evaluate a symbolic expression for a given data row and coefficient values.

- `expr`: An `Expression` (either `ExprLeaf` or `ExprOp`)
- `row`: A NamedTuple or Dict with data (e.g., `(:income => 1000.0)`)
- `β`: A Dict mapping coefficient names to numeric values
"""
function evaluate(expr::Expression, row::NamedTuple, β::Dict{Symbol, Float64})
    if expr isa ExprLeaf
        val = expr.value
        return val isa Variable     ? row[val.name] :
               val isa Coefficient ? β[val.name]    :
               val                 # literal Float64
    elseif expr isa ExprOp
        l = evaluate(expr.left, row, β)
        r = evaluate(expr.right, row, β)
        op = expr.op
        return op == :+ ? l + r :
               op == :- ? l - r :
               op == :* ? l * r :
               op == :/ ? l / r :
               error("Unsupported operator $op")
    else
        error("Invalid expression type")
    end
end