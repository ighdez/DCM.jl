# A simple multinomial logit model
struct MultinomialLogitModel
    utilities::Vector{UtilityFunction}
    choice_variable::Symbol
    data::Any  # For now, a DataFrame or NamedTuple[]
    coefficients::Vector{Coefficient}
end