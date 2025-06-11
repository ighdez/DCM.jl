using Random, Distributions, Sobol, StatsBase

export Draws, generate_draws

"""
Struct to hold simulation draws used for random parameters.

- `values`: Dict with parameter name => matrix (N x R)
- `scheme`: Symbol indicating sampling scheme (:normal, :uniform, :halton, :mlhs)
- `R`: number of draws per individual
"""
struct Draws
    values::Dict
    scheme::Symbol
    R::Int
end

"""
Generates simulation draws for each parameter name provided.

# Arguments
- `param_names`: list of Symbols (e.g., [:time, :cost])
- `N`: number of individuals
- `R`: number of draws
- `scheme`: Symbol indicating sampling scheme (:normal, :uniform, :halton, :mlhs)

# Returns
- `Draws` object containing the generated values
"""
function generate_draws(param_names::Vector{Symbol}, N::Int, R::Int; scheme::Symbol = :normal)
    values = Dict()

    for pname in param_names
        if scheme == :normal
            values[pname] = rand(Normal(), N, R)

        elseif scheme == :uniform
            values[pname] = rand(Uniform(-√3, √3), N, R)

        elseif scheme == :halton
            values[pname] = halton_sequence(N, R, pname)

        elseif scheme == :mlhs
            values[pname] = mlhs_draws(N, R, pname)

        else
            error("Unsupported sampling scheme: $scheme")
        end
    end

    return Draws(values, scheme, R)
end

# Halton sequence generator (using Sobol as a placeholder or to be replaced)
function halton_sequence(N::Int, R::Int, pname::Symbol)
    sobol = SobolSeq(R)
    draws = zeros(N, R)
    for i in 1:N
        draws[i, :] .= 2 .* rand(sobol) .- 1
    end
    return draws
end

# Modified Latin Hypercube Sampling (simplified)
function mlhs_draws(N::Int, R::Int, pname::Symbol)
    draws = zeros(N, R)
    for i in 1:N
        for r in 1:R
            u = (r - 1 + rand()) / R
            draws[i, r] = quantile(Normal(), u)
        end
        shuffle!(draws[i, :])
    end
    return draws
end
