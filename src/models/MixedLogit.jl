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
    utilities::Vector{DCMExpression}          # Utility expressions V_j
    data::DataFrame                           # Dataset
    id::Vector{Int}                           # ID
    availability::Vector{Vector{Bool}}        # Alternative availability
    parameters::Dict                          # Initial parameter values (mu, sigma, etc.)
    draws::Dict                               # Draws: N x R
    draw_scheme::Symbol                       # :normal, :uniform, :mlhs, etc.
    R::Int                                    # Number of simulations (draws)
end

"""
    collect_draws(expr::DCMExpression)

    Recursively traverses a DCMExpression and returns all Draw objects it contains.

    # Arguments
    - `expr`: a symbolic utility expression

    # Returns
    - Vector of `Draw` nodes used in the expression
"""
function collect_draws(expr::DCMExpression)
    nodes = DCMDraw[]

    if expr isa DCMDraw
        push!(nodes, expr)

    elseif expr isa DCMSum || expr isa DCMMult
        append!(nodes, collect_draws(expr.left))
        append!(nodes, collect_draws(expr.right))

    elseif expr isa DCMExp || expr isa DCMMinus
        append!(nodes, collect_draws(expr.arg))

    elseif expr isa DCMEqual
        append!(nodes, collect_draws(expr.left))
        # omit right (it's a Real)

    end

    return nodes
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
    # 1. Collect all Draw objects in the utility expressions
    draw_symbols = Set{Symbol}()
    for u in utilities
        for node in collect_draws(u)
            if node isa DCMDraw
                push!(draw_symbols, node.name)
            end
        end
    end

    # 2. Generate all draws using external Draws.jl infrastructure
    N = nrow(data)
    draw_struct = generate_draws(collect(draw_symbols), N, R; scheme=draw_scheme)
    draws = Dict(s => draw_struct.values[s] for s in keys(draw_struct.values))

    return MixedLogitModel(
        utilities,
        data,
        id,
        availability,
        parameters,
        draws,
        draw_scheme,
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
    availability::Vector{<:AbstractVector{Bool}},
    draws::Dict,
    R::Int,
    id::Vector{Int}
)
    N = nrow(data)
    J = length(utilities)

    # Obtener IDs únicos y crear índice
    unique_ids = unique(id)
    I = length(unique_ids)

    # Validar que la dimensión de draws sea consistente
    # @assert I == size(draws[first(keys(draws))], 1) "Mismatch en número de individuos"

    # Map de ID a índice
    id_index_map = Dict(pid => idx for (idx, pid) in enumerate(unique_ids))

    # Expandir draws a nivel de observación
    expanded_draws = Dict{Symbol, Matrix{Float64}}()
    for (param, matrix) in draws
        _, R_check = size(matrix)
        @assert R == R_check "Mismatch en número de draws"
        param_draws = zeros(N, R)
        for (row_idx, pid) in enumerate(id)
            param_draws[row_idx, :] .= matrix[id_index_map[pid], :]
        end
        expanded_draws[param] = param_draws
    end

    # Evaluar utilidades con draws expandidos
    utils = [evaluate(u, data, parameters, expanded_draws) for u in utilities]

    # Convertir a array 3D: N×R×J
    U = Array{eltype(utils[1]), 3}(undef, N, R, J)
    for j in 1:J
        U[:, :, j] .= utils[j]
    end

    # Aplicar exp a alternativas disponibles
    expU = exp.(clamp.(U, -700, 700))  # previene overflow
    for j in 1:J, n in 1:N
        if !availability[j][n]
            expU[n, :, j] .= 0.0
        end
    end

    # Denominador: suma sobre alternativas
    denom = sum(expU, dims=3)
    denom .= max.(denom, 1e-300)  # failsafe

    # Probabilidades condicionales
    probs_nrj = expU ./ denom

    # Promediar sobre draws
    probs = dropdims(mean(probs_nrj, dims=2), dims=2)
    probs .= max.(probs, 1e-300)

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
        model.draws,
        model.R,
        model.id
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
    return hcat(logit_prob(
        model.utilities,
        model.data,
        results.parameters,
        model.availability,
        model.draws,
        model.R,
        model.id
    ))
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
    probs = predict(model)  # Matrix N x J
    loglik = 0.0
    N = length(choices)
    for n in 1:N
        chosen = choices[n]
        p = probs[n, chosen]
        loglik += log(p)  # failsafe para evitar log(0)
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
        model.utilities;
        data=model.data,
        parameters=full_values,
        availability=model.availability,
        R=model.R,
        draw_scheme=model.draw_scheme,
        id=model.id
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
        Optim.BFGS(),
        Optim.Options(show_trace = verbose, iterations = 1000);
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