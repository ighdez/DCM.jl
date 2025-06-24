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
    parameters::Dict                                # Initial parameter values (mu, sigma, etc.)
    cs_availability::Array                          # Choice set availability
    availability::Array                             # Alternative availability
    expanded_vars::Dict                             # Expanded variables
    expanded_draws::Dict                            # Draws: N x R
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

    # Identify unique individuals
    id = data[:,idvar]

    # Ensure IDs are sorted
    @assert issorted(id) "The vector `id` must be sorted to ensure consistent draw assignment."

    individuals = unique(id)
    count_per_id = countmap(id)

    I = length(individuals)

    # Get máximum CS on the data
    max_C = maximum(values(count_per_id))

    # Collect all Draw and Variable objects in the utility expressions
    draw_symbols = collect_draws(utilities)
    variable_symbols = collect_variables(utilities)
    
    # Generate all draws per individual using external Draws.jl infrastructure
    draw_struct = generate_draws(draw_symbols, I, R; scheme=draw_scheme)
    raw_draws = Dict(s => draw_struct.values[s] for s in draw_symbols)

    # Expand draws to observation level
    expanded_draws = Dict{Symbol, Array{Float64,3}}()
    for (param, matrix) in raw_draws
        param_draws = zeros(I, max_C, R)
        for i in 1:I
            for r in 1:R
                param_draws[i, :, r] .= matrix[i, r]
            end
        end
        expanded_draws[param] = param_draws
    end

    # Expand variables
    expanded_vars = Dict{Symbol, Array{Float64,3}}()

    for var in variable_symbols
        col = data[:, var]
        var_tensor = zeros(I, max_C, R)
        row_idx = 1

        for (i, pid) in enumerate(individuals)
            C_i = count_per_id[pid]
            for c in 1:C_i
                var_tensor[i, c, :] .= col[row_idx]
                row_idx += 1
            end
        end

        expanded_vars[var] = var_tensor
    end

    # Expand availability conditions (I x max_C x R x J)
    J = length(availability)
    expanded_avail = falses(I, max_C, R, J)

    row_idx = 1
    for i in 1:I
        pid = individuals[i]
        C_i = count_per_id[pid]
        for c in 1:C_i
            for j in 1:J
                a = availability[j][row_idx] ? true : false
                expanded_avail[i, c, :, j] .= a
            end
            row_idx += 1
        end
    end

    # Generate CS availability conditions
    cs_avail = falses(I, max_C, R, J)

    for i in 1:I
        pid = individuals[i]
        C_i = count_per_id[pid]
        for c in 1:C_i
            cs_avail[i, c, :, :] .= true
        end
    end

    # Build and return the model
    return MixedLogitModel(
        utilities,
        parameters,
        cs_avail,
        expanded_avail,
        expanded_vars,
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
    parameters::Dict,
    cs_availability::Array,
    availability::Array,
    expanded_vars::Dict,
    expanded_draws::Dict,
)
    J = length(utilities)

    # Evaluated utilities: utils[j] is N × R
    utils = Vector{Array{<:Real,3}}(undef, J)

    Threads.@threads for j in 1:J
        utils[j] = evaluate(utilities[j], parameters, expanded_draws, expanded_vars)
    end

    I, C, R = size(utils[1])

    # Initialize 3D tensor: (N, R, J)
    T = eltype(first(utils))

    # Stack utils into a single tensor U of size (I, C, R, J)
    U = Array{T}(undef, I, C, R, J)

    Threads.@threads for j in 1:J
        @views U[:, :, :, j] .= utils[j]
    end

    # Apply availability → mask entries as -Inf where unavailable
    U .= ifelse.(availability, U, -Inf)

    # Compute probabilities
    expU = exp.(clamp.(U, T(-200), T(200)))              # stabilize
    s_expU = max.(sum(expU, dims=4), T(1e-300))                           # sum across alternatives

    probs = expU ./ max.(s_expU, T(1e-12))

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
        model.parameters,
        model.cs_availability,
        model.availability,
        model.expanded_vars,
        model.expanded_draws,
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
    results.parameters,
    model.cs_availability,
    model.availability,
    model.expanded_vars,
    model.expanded_draws,
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
function loglikelihood(model::MixedLogitModel, choices::Array{Int,3})
    probs = logit_prob(
        model.utilities,
        model.parameters,
        model.cs_availability,
        model.availability,
        model.expanded_vars,
        model.expanded_draws,
    )

    I, C, R, _ = size(probs)
    
    # Initialize simulated probability matrix: R x I
    T = eltype(first(probs))

    log_chosen_probs = zeros(T, I, C, R)

    @inbounds Threads.@threads for i in 1:I
        for c in 1:C
            for r in 1:R
                j = choices[i, c, r]
                if j > 0
                    log_chosen_probs[i, c, r] = log(max(probs[i, c, r, j],T(1e-12)))
                end
            end
        end
    end

    log_indiv_prob = sum(log_chosen_probs, dims=2)  # I × 1 × R
    avg_prob = sum(exp.(log_indiv_prob),dims=3) / model.R
    loglik_i = log.(max.(avg_prob, T(1e-12)))  # I × 1

    return sum(loglik_i)

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

    # Construct Y tensor from cs_availability ∈ Bool[I, C, R, J]
    I, C, R, J = size(model.cs_availability)
    N = length(choicevar)
    Y = zeros(Int, I, C, R)
    idx = 1
    @inbounds for i in 1:I
        for c in 1:C
            if any(model.cs_availability[i, c, :, :])
                Y[i, c, :] .= choicevar[idx]
                idx += 1
            else
                Y[i, c, :] .= 0
            end
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

    mutable_struct = deepcopy(model)

    function f_obj(θ)
        @inbounds for (i, name) in enumerate(free_names)
            mutable_struct.parameters[name] = θ[i]
        end

        @inbounds for name in fixed_names
            mutable_struct.parameters[name] = init_values[name]
        end

        loglik = loglikelihood(mutable_struct,Y)
        return -loglik
    end

    if verbose
        println("Starting optimization routine...")
    end
    
    t_start = time()
    result = Optim.optimize(
        f_obj,
        θ0,
        Optim.BFGS(linesearch = LineSearches.HagerZhang(
            delta = 0.2,           # más conservador que 0.1
            sigma = 0.5,           # curvatura fuerte (evita pasos grandes)
            alphamax = 1.0,        # permite explorar pasos amplios (útil si gradientes son suaves)
            rho = 1e-6,            # mínima diferencia relativa entre pasos
            epsilon = 1e-4,        # precisión media (puede subir si el gradiente es ruidoso)
            gamma = 1e-4,          # estabilidad numérica
            linesearchmax = 30,    # permitir más pasos si gradiente es irregular
            )),
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
    # H = ForwardDiff.hessian(f_obj, θ̂)
    
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