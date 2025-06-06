using DataFrames

struct LogitModel <: DiscreteChoiceModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict
    availability::Vector{<:AbstractVector{Bool}}
end

function LogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    parameters::Dict = Dict(),
    availability::Vector{<:AbstractVector{Bool}} = []
)
    return LogitModel(utilities, data, parameters, availability)
end

function predict(model::LogitModel)
    return logit_prob(model.utilities, model.data, model.parameters, model.availability)
end

function loglikelihood(model::LogitModel, choices::Vector{Int})
    probs = predict(model)  # list of vectors: one per alternative
    loglik = 0.0
    N = length(choices)
    for n in 1:N
        chosen_alt = choices[n]
        chosen_prob = probs[chosen_alt][n]
        loglik += log(chosen_prob)
    end
    return loglik
end

function update_model(model::LogitModel, θ, free_names, fixed_names, init_values)
    full_values = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        full_values[name] = θ[i]
    end
    for name in fixed_names
        full_values[name] = init_values[name]
    end
    return LogitModel(model.utilities;
        data=model.data,
        parameters=full_values,
        availability=model.availability)
end

function estimate(model::LogitModel, choicevar; verbose = true)

    if any(ismissing, choicevar)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choicevar)
    
    params = collect_parameters(model.utilities)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    # Get free/fixed masks
    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    # Initial guess only for free params
    θ0 = [init_values[n] for n in free_names]

    function objective(θ)
        updated = update_model(model, θ, free_names, fixed_names, init_values)

        return -loglikelihood(updated, choices)
    end

    if verbose
        println("Starting optimization routine...")
    end

    t_start = time()
    result = Optim.optimize(objective, θ0, Optim.BFGS(),Optim.Options(show_trace = verbose, iterations = 1000); autodiff = :forward)

    if verbose & Optim.converged(result)
        println("Converged")
    end

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict{Symbol, Real}()

    for (i, name) in enumerate(free_names)
        estimated_params[name] = θ̂[i]
    end

    for name in fixed_names
        estimated_params[name] = init_values[name]
    end

    # Hessian
    if verbose
        println("Computing Standard Errors")
    end

    H = ForwardDiff.hessian(objective, θ̂)
    vcov = inv(H)
    std_errors = sqrt.(diag(vcov))
    t_end = time()

    se = Dict{Symbol, Real}()

    for (i, name) in enumerate(free_names)
        se[name] = std_errors[i]
    end

    return (
        result = result,
        parameters = estimated_params,
        std_errors = se,
        vcov = vcov,
        loglikelihood = -Optim.minimum(result),
        iters = Optim.iterations(result),
        converged = Optim.converged(result),
        estimation_time = t_end - t_start
    )
end