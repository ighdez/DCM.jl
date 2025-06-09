# using DataFrames

struct MixedLogitModel <: DiscreteChoiceModel
    utilities::Vector{<:DCMExpression}        # Symbolic utility functions
    data::DataFrame                            # Data for evaluation
    availability::Vector{<:AbstractVector{Bool}} # Availability of alternatives
    parameters::Dict             # Initial/fixed parameter values
    draws::Draws                               # Simulation draws
end

"""
Constructor for MixedLogitModel.
Automatically infers required draws from utility expressions.

# Arguments
- `utilities`: Vector of symbolic utility expressions
- `data`: DataFrame
- `parameters`: Dict of initial/fixed parameter values
- `availability`: Vector of Bool vectors
- `R`: Number of simulation draws
- `draw_scheme`: Symbol (:normal, :uniform, :halton, :mlhs)

# Returns
- `MixedLogitModel` instance
"""
function MixedLogitModel(utilities::Vector{<:DCMExpression};
                         data::DataFrame,
                         parameters::Dict=Dict(),
                         availability::Vector{<:AbstractVector{Bool}}=[],
                         R::Int=100,
                         draw_scheme::Symbol=:mlhs)

    # Infer required draw names from utility expressions
    draw_names = Symbol[]
    function find_draws(expr)
        if expr isa DCMDraw
            push!(draw_names, expr.name)
        elseif expr isa DCMBinary
            find_draws(expr.left)
            find_draws(expr.right)
        elseif expr isa DCMUnary
            find_draws(expr.arg)
        end
    end
    for u in utilities
        find_draws(u)
    end
    draw_names = unique(draw_names)

    # Generate draws for inferred parameters
    N = nrow(data)
    draw_obj = generate_draws(draw_names, N, R; scheme=draw_scheme)

    return MixedLogitModel(utilities, data, availability, parameters, draw_obj)
end

"""
Predicts choice probabilities by averaging over draws.
Returns a matrix of size N x J
"""
function predict(model::MixedLogitModel)
    N = nrow(model.data)
    R = model.draws.R
    J = length(model.utilities)

    probs_draws = Array{Float64, 3}(undef, N, J, R)

    for (j, uj) in enumerate(model.utilities)
        utils_j = evaluate(uj, model.data, model.parameters, model.draws.values)  # N x R
        if !isempty(model.availability)
            avail_j = model.availability[j]
            utils_j = utils_j .* reshape(avail_j, N, 1)
        end
        probs_draws[:, j, :] .= exp.(utils_j)
    end

    denom = sum(probs_draws, dims=2)  # N x 1 x R
    probs = probs_draws ./ denom      # N x J x R

    mean_probs = mean(probs, dims=3)[:, :, 1]  # N x J
    return mean_probs
end

"""
Computes simulated log-likelihood of a MixedLogitModel.

# Arguments:
- model: a MixedLogitModel instance
- choices: vector of integers with chosen alternatives (1-based)

# Returns:
- Total simulated log-likelihood
"""
function loglikelihood(model::MixedLogitModel, choices::Vector{Int})
    N = nrow(model.data)
    R = model.draws.R
    J = length(model.utilities)

    probs_draws = Array{Float64, 3}(undef, N, J, R)

    for (j, uj) in enumerate(model.utilities)
        utils_j = evaluate(uj, model.data, model.parameters, model.draws.values)
        if !isempty(model.availability)
            avail_j = model.availability[j]
            utils_j = utils_j .* reshape(avail_j, N, 1)
        end
        probs_draws[:, j, :] .= exp.(utils_j)
    end

    denom = sum(probs_draws, dims=2)
    probs = probs_draws ./ denom  # N x J x R

    # Extract probabilities of chosen alternatives
    loglik = 0.0
    for n in 1:N
        chosen = choices[n]
        p_avg = mean(probs[n, chosen, :])
        loglik += log(p_avg)
    end

    return loglik
end

function update_model(model::MixedLogitModel, θ, free_names, fixed_names, init_values)
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
Estimates the parameters of a MixedLogitModel using MLE over simulated log-likelihood.
"""
function estimate(model::MixedLogitModel, choicevar; verbose=true)
    # using Optim, ForwardDiff

    if any(ismissing, choicevar)
        error("Choice vector contains missing values. Please clean your data.")
    end

    choices = Int.(choicevar)

    param_names = collect(keys(model.parameters))
    θ0 = [model.parameters[n] for n in param_names]

    function objective(θ)
        param_dict = Dict{Symbol, Real}()
        for (i, name) in enumerate(param_names)
            param_dict[name] = θ[i]
        end
        updated = MixedLogitModel(model.utilities;
                                  data=model.data,
                                  parameters=param_dict,
                                  availability=model.availability,
                                  R=model.draws.R,
                                  draw_scheme=model.draws.scheme)
        return -loglikelihood(updated, choices)
    end

    if verbose
        println("Starting optimization routine...")
    end

    t_start = time()
    result = Optim.optimize(objective, θ0, Optim.BFGS(), Optim.Options(show_trace = verbose, iterations = 1000); autodiff = :forward)

    if verbose && Optim.converged(result)
        println("Converged")
    end

    θ̂ = Optim.minimizer(result)
    estimated_params = Dict{Symbol, Real}()
    for (i, name) in enumerate(param_names)
        estimated_params[name] = θ̂[i]
    end

    if verbose
        println("Computing Standard Errors")
    end

    H = ForwardDiff.hessian(objective, θ̂)
    vcov = inv(H)
    std_errors = sqrt.(diag(vcov))
    t_end = time()

    se = Dict{Symbol, Float64}()
    for (i, name) in enumerate(param_names)
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
