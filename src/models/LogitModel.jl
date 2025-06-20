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

    # Initialize matrix (N x J)
    T = eltype(first(utils))
    expU = zeros(T,N,J)
    s_expU = zeros(T,N)
    probs = zeros(T,N,J)

    # Loop over rows
    Threads.@threads for n in 1:N
            for j in 1:J
                U_nj = utils[j][n]
                av_nj = availability[j][n]
                expU[n,j] = av_nj ? exp(U_nj) : T(0.0)
                s_expU[n] += expU[n,j]
            end
            for j in 1:J
                probs[n,j] = expU[n,j] / s_expU[n]
            end
        end

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
        loglik[n] = log(p)
    end

    return sum(loglik)
end

function gradient(model::LogitModel, choices::Vector{Int}, dU::Vector{Vector{DCMExpression}})
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        model.parameters,
    )
    
    N, J = size(probs)
    K = length(dU[1])

    T = eltype(first(probs))
    grad = zeros(T,K)
    dU_vals = [Vector{Vector{T}}(undef, J) for _ in 1:K]

    # Evaluate derivatives once
    Threads.@threads for k in 1:K
        for j in 1:J
            dU_vals[k][j] = evaluate(dU[j][k], model.data, model.parameters)
        end
    end

    # Loop over obs
    grad_vecs = [zeros(T, N) for _ in 1:K]
    Threads.@threads for n in 1:N
        chosen = choices[n]
        for k in 1:K
            ∂U_ch = dU_vals[k][chosen][n]
            s = zero(eltype(model.data[!,1]))
            for j in 1:J
                s += probs[n,j] * dU_vals[k][j][n]
            end
            grad_vecs[k][n] = ∂U_ch - s
        end
    end

    for k in 1:K
        grad[k] = sum(grad_vecs[k])
    end

    return grad
end

"""
function update_model(model::LogitModel, θ, free_names, fixed_names, init_values)

Updates a LogitModel with a new set of parameter values during optimization.

# Arguments

* `model`: current `LogitModel`
* `θ`: vector of values for free parameters
* `free_names`: names of free parameters
* `fixed_names`: names of fixed parameters
* `init_values`: original values used to complete full parameter vector

# Returns

Updated `LogitModel` instance
"""
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

    # Precompute derivatives
    dU_tmp = [[derivative(u, pname) for pname in free_names] for u in model.utilities]

    # Force type
    dU = [convert(Vector{DCMExpression}, dj) for dj in dU_tmp]

    function f_obj(θ)
        for (i, name) in enumerate(free_names)
            mutable_struct.parameters[name] = θ[i]
        end

        for name in fixed_names
            mutable_struct.parameters[name] = init_values[name]
        end

        loglik = loglikelihood(mutable_struct,choices)
        return -loglik
    end

    function g_obj(θ)
        for (i, name) in enumerate(free_names)
            mutable_struct.parameters[name] = θ[i]
        end

        for name in fixed_names
            mutable_struct.parameters[name] = init_values[name]
        end

        grad = gradient(mutable_struct,choices,dU)
        return -grad
    end

    if verbose
        println("Starting optimization routine...")
    end

    t_start = time()
    result = Optim.optimize(
            f_obj,
            g_obj,
            θ0,
            Optim.BFGS(),
            Optim.Options(
                show_trace = verbose,
                iterations = 1000),inplace=false)

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

    H = FiniteDiff.finite_difference_hessian(f_obj, θ̂)
    
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