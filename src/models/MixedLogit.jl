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
    id::Tuple{Dict,Vector}                          # ID
    availability::Vector{Vector{Bool}}              # Alternative availability
    parameters::Dict                                # Initial parameter values (mu, sigma, etc.)
    draws::Dict                                     # Draws: N x R
    R::Int                                          # Number of simulations (draws)
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
    idvar::Symbol,
    availability::Vector{<:AbstractVector{Bool}} = [],
    parameters::Dict = Dict(),
    draw_scheme::Symbol = :normal,
    R::Int = 100
)

    # Get id variable
    id = data[:,idvar]

    # 0. Ensure IDs are sorted
    @assert issorted(id) "The vector `id` must be sorted to ensure consistent draw assignment."

    # 1. Collect all Draw objects in the utility expressions
    draw_symbols = collect_draws(utilities)

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

    # 5. Build and return the model
    return MixedLogitModel(
        utilities,
        data,
        (id_index_map, id),
        availability,
        parameters,
        expanded_draws,
        R
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
    parameters::Dict,
    availability::Vector{<:AbstractVector{Bool}},  # N × J
    draws::Dict{Symbol, Matrix{Float64}}, # N × R
)
    J = length(utilities)

    # Evaluated utilities: utils[j] is N × R
    utils = Vector{Matrix{<:Real}}(undef, J)

    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], data, parameters, draws)
    end

    N, R = size(utils[1])

    # Initialize 3D tensor: (N, J, R)
    T = eltype(first(utils))

    # Stack utils into a single tensor U of size (N, J, R)
    expU = Array{T}(undef, N, J, R)
    s_expU = Array{T}(undef, N, R)

    Threads.@threads for r in 1:R
        @inbounds begin
            for j in 1:J
                # u = clamp.(utils[j][:,r], T(-200), T(200))
                u = utils[j][:,r]
                expU[:, j, r] .= ifelse.(availability[j], exp.(u), 0.0)
            end
            s_expU[:, r] .= sum(expU[:, :, r]; dims = 2)
        end
    end
    @inbounds probs = expU ./ max.(reshape(s_expU, N, 1, R), T(1e-30))

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
        model.parameters,
        model.availability,
        model.draws
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
        results.parameters,
        model.availability,
        model.draws
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
function loglikelihood(model::MixedLogitModel, Y::Array{Bool,3};parameters::Dict=mutable_parameters)
    probs = logit_prob(
        model.utilities,
        model.data,
        parameters,
        model.availability,
        model.draws
)

    N, _, R = size(probs)
    id_map, id = model.id
    I = length(id_map)
    
    # Initialize simulated probability matrix: R x I
    T = eltype(first(probs))

    # Compute log-probabilities with failsafe
    log_indiv = zeros(T, I, R)
    indiv_prob = Array{T}(undef, I, R)
    loglik = zeros(T, I)
    Threads.@threads for r in 1:R
        @inbounds begin
            log_probs = log.(max.(probs[:, :, r], T(1e-30)))      # N × J
            log_chosen = sum(log_probs .* Y[:, :, r]; dims = 2)  # N × 1
            for n in 1:N
                i = id_map[id[n]]
                log_indiv[i, r] += log_chosen[n, 1]  # extrae escalar
            end
            indiv_prob[:, r] .= exp.(log_indiv[:, r])  # ← CORRECTO
        end
    end

    # Final log-likelihood with failsafe
    Threads.@threads for i in 1:I
        @inbounds begin
            avg_prob = sum(indiv_prob[i, :]) / R
            loglik[i] = log(max(avg_prob, T(1e-30)))
        end
    end

    return loglik
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
function estimate(model::MixedLogitModel, choicevar::Symbol; verbose::Bool = true)
    
    choice_data = model.data[:,choicevar]

    if any(ismissing, choice_data)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choice_data)

    # Construct Y tensor (one-hot encoding) from cs_availability
    J = length(model.utilities)
    N, R = size(first(values(model.draws)))

    Y = zeros(Bool, N, J, R)
    @inbounds for n in 1:N
        j = choices[n]
        if j > 0
            Y[n, j, :] .= true
        end
    end

    params = collect_parameters(model.utilities)
    param_names = [p.name for p in params]
    init_values = Dict(p.name => p.value for p in params)
    is_fixed = [p.fixed for p in params]

    # Get free/fixed masks
    free_names = param_names[.!is_fixed]
    fixed_names = param_names[is_fixed]

    # Initial guess only for free params
    θ0 = [init_values[n] for n in free_names]

    # Preallocate mutable parameter set (no deepcopy of full model)
    mutable_parameters = deepcopy(model.parameters)

    function f_obj_i(θ)
        @inbounds begin
            for (i, name) in enumerate(free_names)
                mutable_parameters[name] = θ[i]
            end
            
            for name in fixed_names
                mutable_parameters[name] = init_values[name]
            end
        end

        loglik = loglikelihood(model, Y; parameters=mutable_parameters)
        return -loglik
    end

    function f_obj(θ)
        @inbounds begin
            for (i, name) in enumerate(free_names)
                mutable_parameters[name] = θ[i]
            end
            
            for name in fixed_names
                mutable_parameters[name] = init_values[name]
            end
        end

        loglik = loglikelihood(model, Y; parameters=mutable_parameters)
        return -sum(loglik)
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
        Optim.BFGS(linesearch=LineSearches.BackTracking()),#linesearch = LineSearches.HagerZhang(
            # delta = 0.2,           # más conservador que 0.1
            # sigma = 0.8,           # curvatura fuerte (evita pasos grandes)
            # alphamax = 1.0,        # permite explorar pasos amplios (útil si gradientes son suaves)
            # rho = 1e-6,            # mínima diferencia relativa entre pasos
            # epsilon = 1e-4,        # precisión media (puede subir si el gradiente es ruidoso)
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

    ForwardDiff.hessian!(H, f_obj, θ̂, cfg)

    vcov = try 
            inv(H)
        catch
            pinv(H)
    end
    
    std_errors = sqrt.(diag(vcov))

    se = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        se[name] = std_errors[i]
    end

    if verbose
        println("Computing Robust Standard Errors")
    end
    scores = ForwardDiff.jacobian(f_obj_i, θ̂)  # N × K
    G = scores' * scores  # K × K    

    V_rob = try
        inv(H) * G * inv(H)
    catch
        pinv(H) * G * pinv(H)
    end

    rob_std_errors = sqrt.(diag(V_rob))

    rob_se = Dict{Symbol, Real}()
    for (i, name) in enumerate(free_names)
        rob_se[name] = rob_std_errors[i]
    end

    t_end = time()
    return (
        result = result,
        parameters = estimated_params,
        std_errors = se,
        vcov = vcov,
        rob_std_errors = rob_se,
        rob_vcov = V_rob,
        loglikelihood = -Optim.minimum(result),
        iters = Optim.iterations(result),
        converged = Optim.converged(result),
        estimation_time = t_end - t_start
    )
end