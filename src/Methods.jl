import ..DCM: DCMExpression, DCMParameter, DCMVariable, DCMSum, DCMMult, DCMExp

export evaluate

function evaluate(expr::DCMExpression, data::Dict{Symbol, Vector{Float64}}, params::Dict{Symbol, Float64})
    if expr isa DCMParameter
        return fill(params[expr.name], length(first(values(data))))
    elseif expr isa DCMVariable
        return data[expr.name]
    elseif expr isa DCMSum
        return evaluate(expr.left, data, params) .+ evaluate(expr.right, data, params)
    elseif expr isa DCMMult
        return evaluate(expr.left, data, params) .* evaluate(expr.right, data, params)
    elseif expr isa DCMExp
        return exp.(evaluate(expr.arg, data, params))
    else
        error("Unknown expression type")
    end
end
