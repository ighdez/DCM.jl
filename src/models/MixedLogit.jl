"""
Implementation of the MixedLogitModel type and estimation methods.

This module defines the MixedLogitModel struct and all associated functions required to estimate, update, and predict a mixed multinomial logit model. Built on top of the symbolic utilities defined in Expressions.jl.
"""

using DataFrames, StatsBase

"""
    MixedLogitModel

    Structure for a mixed logit model (Mixed Logit). It allows defining symbolic utility functions
    that include fixed and random parameters (represented by `Draw`). Automatically generates the
    required draws for random coefficients during model construction.

    # Fields
    - `utilities::Vector{DCMExpression}`: List of utility expressions for each alternative.
    - `data::DataFrame`: Dataset with individual choice observations.
    - `id::Vector{Int}`: ID variable
    - `availability::Vector{Vector{Bool}}`: Availability of each alternative per observation.
    - `parameters::Dict{Symbol, Float64}`: Dictionary with initial parameter values.
    - `draws::Dict{Symbol, Matrix{Float64}}`: Dictionary of draws, with matrices of size (R x N).
    - `draw_scheme::Symbol`: Drawing scheme (`:normal`, `:uniform`, `:mlhs`, etc.).
    - `R::Int`: Number of simulations per individual.
"""

struct MixedLogitModel <: DiscreteChoiceModel
    utilities::Vector{DCMExpression}                # Utility expressions V_j
    data::DataFrame                                 # Dataset
    id::Vector{Int}                                 # ID
    availability::Vector{Vector{Bool}}              # Alternative availability
    parameters::Dict                                # Initial parameter values (mu, sigma, etc.)
    draws::Dict                                     # Draws: N x R
    draw_scheme::Symbol                             # :normal, :uniform, :mlhs, etc.
    R::Int                                          # Number of simulations (draws)
    expanded_vars::Dict                             # Expanded variables
end

"""
    MixedLogitModel(utilities; data, availability, parameters, R, draw_scheme)

    Constructs a `MixedLogitModel` object from symbolic utility expressions. It automatically detects
    all `Draw` objects used in the model, generates the required simulation draws, and stores all
    necessary data for estimation and prediction.

    See also: `generate_draws`, `Draw`, `Parameter`, `Variable`

    # Arguments
    - `utilities::Vector{DCMExpression}`: Utility expressions for each alternative.
    - `data::DataFrame`: Dataset.
    - `availability::Vector{Vector{Bool}}` (optional): Availability indicators.
    - `parameters::Dict` (optional): Initial parameter values.
    - `R::Int` (optional): Number of simulations.
    - `draw_scheme::Symbol` (optional): Drawing scheme (e.g., `:normal`, `:mlhs`).

    # Returns
    - `MixedLogitModel`: A model object ready for estimation or prediction.
"""
function MixedLogitModel(
    utilities::Vector{<:DCMExpression};
    data::DataFrame,
    id::Vector{Int},
    availability::Vector{<:AbstractVector{Bool}} = [],
    parameters::Dict = Dict(),
    draw_scheme::Symbol = :normal,
    R::Int = 100
)

    # 0. Ensure IDs are sorted
    @assert issorted(id) "The vector `id` must be sorted to ensure consistent draw assignment."

    # 1. Collect all Draw objects in the utility expressions
    draw_symbols = collect_draws(utilities)
    variable_symbols = collect_variables(utilities)

    # 2. : Identify unique individuals
    individuals = unique(id)
    N_individuals = length(individuals)
    
    # 3. Generate all draws per individual using external Draws.jl infrastructure
    draw_struct = generate_draws(draw_symbols, N_individuals, R; scheme=draw_scheme)
    raw_draws = Dict(s => draw_struct.values[s] for s in draw_symbols)

    # 4. Expand draws to observation level
    id_index_map = Dict(pid => idx for (idx, pid) in enumerate(individuals))
    N = nrow(data)
    expanded_draws = Dict{Symbol, Matrix{Float64}}()
    for (param, matrix) in raw_draws
        param_draws = zeros(N, R)
        for (i, pid) in enumerate(id)
            param_draws[i, :] .= matrix[id_index_map[pid], :]
        end
        expanded_draws[param] = param_draws
    end

    # 5. Expand variables
    expanded_vars = Dict{Symbol, Matrix{Float64}}()
    for var in variable_symbols
        col = data[:, var]
        expanded_vars[var] = repeat(reshape(col, N, 1), 1, R)
    end

    # 5. Build and return the model
    return MixedLogitModel(
        utilities,
        data,
        id,
        availability,
        parameters,
        expanded_draws,
        draw_scheme,
        R,
        expanded_vars
    )
end

"""
    logit_prob(utilities, data, parameters, availability, draws, R)

    Computes simulated choice probabilities for a Mixed Logit Model using R draws.

    # Returns
    - Matrix{Float64} of size (N, J): Simulated choice probabilities.
"""

function logit_prob(
    utilities::Vector{<:DCMExpression},
    data::DataFrame,
    availability::Vector{<:AbstractVector{Bool}},
    parameters::Dict,
    draws::Dict,
    expanded_vars::Dict,
    R::Int,
)
    N = nrow(data)
    J = length(utilities)

    # Evaluate utility for each alternative -> utils[j] is N x R
    utils = Vector{Matrix}(undef, J)
    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], data, parameters, draws, expanded_vars)
    end
    # utils = [evaluate(u, data, parameters, draws, expanded_vars) for u in utilities]

    # Initialize 3D tensor: (N, R, J)
    T = eltype(first(utils))
    U = Array{T, 3}(undef, N, R, J)

    Threads.@threads for j in 1:J
        U[:, :, j] .= utils[j]  # Each slice is utility of alt j across obs x draws
    end

    # Compute exp(U), applying availability constraints and failsafe
    # U = clamp.(U, -20, 20)
    # expU = exp.(clamp.(U, -100.0, 100.0))  # Avoid overflow
    expU = clamp.(exp.(U), 1e-30, 1e+30) # Avoid overflow

    Threads.@threads for j in 1:J
        for n in 1:N
            if !availability[j][n]
                expU[n, :, j] .= 0.0
            end
        end
    end

    # Compute denominator and normalize
    denom = sum(expU, dims=3)
    denom .= max.(denom, 1e-300)  # Numerical failsafe

    probs_nrj = expU ./ denom  # (N, R, J)
    return probs_nrj

end

"""
    predict(model::MixedLogitModel)

    Computes predicted choice probabilities for each alternative using simulated draws.

    # Returns
    - Matrix{Float64} of size (N, J): Simulated choice probabilities.
"""
function predict(model::MixedLogitModel)
    return logit_prob(
        model.utilities,
        model.data,
        model.availability,
        model.parameters,
        model.draws,
        model.expanded_vars,
        model.R,
    )
end

"""
    predict(model::MixedLogitModel, results)

    Computes predicted probabilities using estimated parameters from `results`.

    # Arguments
    - `model`: a `MixedLogitModel` instance
    - `results`: a named tuple with at least a `parameters` field

    # Returns
    - Matrix{Float64} of size (N, J): Simulated choice probabilities
"""
function predict(model::MixedLogitModel, results)
    return logit_prob(
        model.utilities,
        model.data,
        model.availability,
        results.parameters,
        model.draws,
        model.expanded_vars,
        model.R,
    )
end

"""
    loglikelihood(model::MixedLogitModel, choices::Vector{Int})

    Computes the log-likelihood of a Mixed Logit Model given observed choices.

    # Arguments
    - `model`: a `MixedLogitModel` instance
    - `choices`: vector of integers representing chosen alternatives

    # Returns
    - Total log-likelihood value (Float64)
"""
function loglikelihood(model::MixedLogitModel, choices::Vector{Int})
    probs = logit_prob(
        model.utilities,
        model.data,
        model.availability,
        model.parameters,
        model.draws,
        model.expanded_vars,
        model.R
    )

    N, R, _ = size(probs)

    # Unique individuals and ID map
    unique_ids = unique(model.id)
    I = length(unique_ids)
    id_map = Dict(pid => i for (i, pid) in enumerate(unique_ids))

    # Initialize simulated probability matrix: R x I
    T = eltype(probs)
    # indiv_prob = ones(T, R, I)
    log_indiv_prob = zeros(T, R, I)
    # indiv_prob = ones(R, I)

    # Multiply probabilities of chosen alternatives across observations for each individual and draw
    Threads.@threads for n in 1:N
        chosen = choices[n]
        i = id_map[model.id[n]]
        for r in 1:R
            p = probs[n, r, chosen]
            log_indiv_prob[r, i] += log(max(p,1e-12))
        end
    end

    # Average over draws and compute log-likelihood
    loglik = 0.0
    indiv_prob = exp.(log_indiv_prob)
    for i in 1:I
        mean_prob = mean(indiv_prob[:, i])
        loglik += log(max(mean_prob, 1e-12))  # failsafe
    end

    return loglik
end

"""
    update_model(model::MixedLogitModel, θ, free_names, fixed_names, init_values)

    Returns a new `MixedLogitModel` with updated parameters while preserving draws and structure.

    # Arguments
    - `model`: current `MixedLogitModel`
    - `θ`: vector of new values for free parameters
    - `free_names`: names of free parameters
    - `fixed_names`: names of fixed parameters
    - `init_values`: dictionary of all initial parameter values

    # Returns
    - Updated `MixedLogitModel` instance
"""
function update_model(model::MixedLogitModel, θ, free_names, fixed_names, init_values)
    full_values = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        full_values[name] = θ[i]
    end
    for name in fixed_names
        full_values[name] = init_values[name]
    end

    return MixedLogitModel(
        model.utilities,
        model.data,
        model.id,
        model.availability,
        full_values,
        model.draws,
        model.draw_scheme,
        model.R,
        model.expanded_vars
    )
end

"""
    estimate(model::MixedLogitModel, choicevar; verbose = true)

    Estimates the parameters of a `MixedLogitModel` via simulated maximum likelihood using `Optim.jl`.

    # Arguments
    - `model`: a `MixedLogitModel` instance
    - `choicevar`: vector of integers indicating the chosen alternatives
    - `verbose`: whether to print status messages (default: true)

    # Returns
    - Named tuple with estimation results:
        - `parameters`: estimated parameter values
        - `std_errors`: standard errors for free parameters
        - `vcov`: variance-covariance matrix
        - `loglikelihood`: final log-likelihood value
        - `iters`: number of iterations
        - `converged`: convergence status
        - `estimation_time`: total time in seconds
"""
function estimate(model::MixedLogitModel, choicevar; verbose = true)
    
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
    result = Optim.optimize(
        objective,
        θ0,
        Optim.BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(
            show_trace = verbose,
            iterations = 1000,
            f_abstol=1e-6,
            g_abstol=1e-8);
        autodiff = :forward
    )

    if verbose && Optim.converged(result)
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

    if verbose
        println("Computing Standard Errors")
    end

    H = ForwardDiff.hessian(objective, θ̂)
    # H = FiniteDiff.finite_difference_hessian(objective, θ̂)
    # vcov = pinv(H)
    std_errors = sqrt.(diag(H \ I))
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