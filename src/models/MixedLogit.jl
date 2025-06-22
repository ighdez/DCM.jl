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
    id_dict::Dict                                   # Unique ID dict
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
        expanded_vars,
        id_index_map
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

    # Initialize 3D tensor: (N, R, J)
    T = eltype(first(utils))
    expU = Array{T, 3}(undef, N, R, J)
    s_expU = zeros(T,N,R)
    probs = zeros(T,N,R,J)

    # Loop over rows
    Threads.@threads for n in 1:N
        for r in 1:R
            for j in 1:J
                U_nrj = utils[j][n,r]
                av_nj = availability[j][n]
                expU[n,r,j] = av_nj ? exp(clamp(U_nrj,T(-200.0),T(200.0))) : T(0)
                s_expU[n,r] += expU[n,r,j]
            end
            s_expU[n,r] = max(s_expU[n,r],T(1e-300)) # Failsafe to avoid divide by zero
            for j in 1:J
                probs[n,r,j] = expU[n,r,j] / s_expU[n,r]
            end
        end
    end
return probs
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
    id_map = model.id_dict
    I = length(id_map)
    
    # Initialize simulated probability matrix: R x I
    T = eltype(probs)
    log_indiv_prob = zeros(T, R, I)
    indiv_prob = zeros(T, I)
    loglik = zeros(T, I)


    # Multiply probabilities of chosen alternatives across observations for each individual and draw
    Threads.@threads for n in 1:N
        chosen = choices[n]
        i = id_map[model.id[n]]
        for r in 1:R
            p = probs[n, r, chosen]
            log_indiv_prob[r, i] += log(max(p,T(1e-12)))
        end
    end

    # Average over draws and compute log-likelihood
    Threads.@threads for i in 1:I
        for r in 1:R
            indiv_prob[i] += max(exp(log_indiv_prob[r,i]),T(1e-12))
        end
        @inbounds indiv_prob[i] = indiv_prob[i] / R
        @inbounds loglik[i] = log(max(indiv_prob[i],T(1e-12)))
    end
    return sum(loglik)
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
        println("Starting optimization routine...")
    end
    
    t_start = time()
    result = Optim.optimize(
        f_obj,
        θ0,
        Optim.BFGS(),#linesearch = LineSearches.HagerZhang(
            # delta = 0.2,           # más conservador que 0.1
            # sigma = 0.8,           # curvatura fuerte (evita pasos grandes)
            # alphamax = 2.0,        # permite explorar pasos amplios (útil si gradientes son suaves)
            # rho = 1e-10,           # mínima diferencia relativa entre pasos
            # epsilon = 1e-6,        # precisión media (puede subir si el gradiente es ruidoso)
            # gamma = 1e-4,          # estabilidad numérica
            # linesearchmax = 30,    # permitir más pasos si gradiente es irregular
            # )),
            Optim.Options(
            show_trace = verbose,
            iterations = 1000,
            f_abstol=1e-6,
            g_abstol=1e-8);
            autodiff=:forward
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