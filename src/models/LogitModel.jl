"""
Implementation of the LogitModel type and estimation methods.

This module defines the LogitModel struct and all associated functions required to estimate, update, and predict a multinomial logit model. Built on top of the symbolic utilities defined in Expressions.jl.
"""

using DataFrames

"""
LogitModel

Data structure for multinomial logit models.

Fields:

* `utilities`: Vector of symbolic utility expressions (one per alternative)
* `data`: DataFrame containing the input data
* `parameters`: Dictionary of parameter values
* `availability`: Vector of boolean vectors indicating alternative availability per observation
  """
struct LogitModel <: DiscreteChoiceModel
    utilities::Vector{<:DCMExpression}
    data::DataFrame
    parameters::Dict
    availability::Vector{<:AbstractVector{Bool}}
end


"""
function LogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    parameters::Dict = Dict(),
    availability::Vector{<:AbstractVector{Bool}} = []
)

Constructor for `LogitModel`.

# Arguments

* `utilities`: vector of symbolic utility expressions
* `data`: DataFrame with data used for evaluation
* `parameters`: dictionary with initial/fixed parameter values (optional)
* `availability`: availability flags per alternative (optional)

# Returns

A `LogitModel` instance
"""
function LogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    parameters::Dict = Dict(),
    availability::Vector{<:AbstractVector{Bool}} = []
)
    return LogitModel(utilities, data, parameters, availability)
end

"""
Computes choice probabilities for the Multinomial Logit model.

# Arguments

* `utilities`: vector of symbolic utility expressions (one per alternative)
* `data`: DataFrame of input data
* `parameters`: parameter dictionary with values for evaluation
* `availability`: vector of boolean vectors indicating available alternatives

# Returns

A vector of vectors, each inner vector representing choice probabilities for each alternative per observation.
"""
function logit_prob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    availability::Vector{<:AbstractVector{Bool}},
    parameters::Dict
)
    N = nrow(data)         # Number of observations
    J = length(utilities)  # Number of alternatives
    
    utils = Vector{Vector}(undef,J)
    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], data, parameters)
    end

    # Initialize arrays
    T = eltype(first(utils))
    
    # Utilities
    U = Array{T}(undef, N, J)
    @inbounds for j in 1:J
        @views U[:, j] .= utils[j]
    end
    
    # Apply availability conditions
    @inbounds for j in 1:J
        @views U[:, j] .= ifelse.(availability[j], U[:, j], -Inf)
    end
    
    # Calculate choice probabilities
    expU = exp.(U)
    s_expU = sum(expU, dims=2)
    probs = expU ./ s_expU

    return probs
end

"""
function predict(model::LogitModel)

Computes predicted probabilities for each alternative using the current parameters in the model.

# Arguments

* `model`: a `LogitModel` instance

# Returns

A vector of vectors with choice probabilities
"""
function predict(model::LogitModel)
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        model.parameters,
    )
    return probs
end


"""
function predict(model::LogitModel,results)

Computes predicted probabilities using estimated parameters.

# Arguments

* `model`: a `LogitModel` instance
* `results`: a named tuple returned by `estimate`

# Returns

A matrix of size N x J, where N is number of observations, J number of alternatives
"""
function predict(model::LogitModel,results)
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        results.parameters,
    )
    return probs
end


"""
function loglikelihood(model::LogitModel, choices::Vector{Int})

Computes the log-likelihood value of the model given observed choices.

# Arguments

* `model`: `LogitModel` instance
* `choices`: vector of integers representing chosen alternatives

# Returns

Total log-likelihood value
"""
function loglikelihood(model::LogitModel, choices::Vector{Int})
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        model.parameters,
    )

    N, _ = size(probs)

    T = eltype(first(probs))
    loglik = zeros(T, N)

    # Loop over observations
    Threads.@threads for n in 1:N
        chosen = choices[n]
        p = probs[n, chosen]
        @inbounds loglik[n] = log(p)
    end

    return sum(loglik)
end

"""
function estimate(model::LogitModel, choicevar; verbose = true)

Estimates model parameters using maximum likelihood estimation.
Uses `Optim.jl` to minimize the negative log-likelihood.

# Arguments

* `model`: a `LogitModel` instance
* `choicevar`: vector of chosen alternatives
* `verbose`: whether to print optimization status (default: true)

# Returns

Named tuple with results, including:

* `parameters`: estimated parameters
* `std_errors`: standard errors
* `vcov`: variance-covariance matrix
* `loglikelihood`: log-likelihood value
* `iters`: number of iterations
* `converged`: convergence status
* `estimation_time`: total time in seconds
  """
function estimate(
    model::LogitModel,
    choicevar::Vector{Int};
    verbose::Bool = true
)

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

    mutable_struct = deepcopy(model)

    function f_obj(θ)
        @inbounds for (i, name) in enumerate(free_names)
            mutable_struct.parameters[name] = θ[i]
        end

        @inbounds for name in fixed_names
            mutable_struct.parameters[name] = init_values[name]
        end

        loglik = loglikelihood(mutable_struct,choices)
        return -loglik
    end

    if verbose
        println("Warming-up hessian...")
    end
    # Warm-up hessian
    H = zeros(length(θ0), length(θ0))
    cfg = ForwardDiff.HessianConfig(f_obj, θ0)
    H = ForwardDiff.hessian!(H, f_obj, θ0, cfg)

    if verbose
        println("Starting optimization routine...")
    end

    t_start = time()
    result = Optim.optimize(
            f_obj,
            θ0,
            Optim.BFGS(),
            Optim.Options(
                show_trace = verbose,
                iterations = 1000);autodiff=:forward)

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

    ForwardDiff.hessian!(H, f_obj, θ̂, cfg)
    
    vcov = try
        inv(H)
    catch
        pinv(H)
    end

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