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
            values[pname] = halton_draws(N, R, pname)

        elseif scheme == :mlhs
            values[pname] = mlhs_draws(N, R, pname)

        else
            error("Unsupported sampling scheme: $scheme")
        end
    end

    return Draws(values, scheme, R)
end

# Halton sequence generator (using Sobol as a placeholder or to be replaced)
function halton_draws(N::Int, R::Int, pname::Symbol)
    function halton(n, base)
        f, r = 1.0, 0.0
        while n > 0
            f /= base
            r += f * (n % base)
            n ÷= base
        end
        return r
    end

    primes = Primes.primes(100)
    index = hash(pname) % length(primes) + 1
    base = primes[index]

    draws = zeros(N, R)
    for i in 1:N
        for r in 1:R
            u = halton((i - 1) * R + r, base)
            draws[i, r] = quantile(Normal(), u)
        end
    end
    return draws
end

# Modified Latin Hypercube Sampling
function mlhs_draws(N::Int, R::Int, pname::Symbol)
    draws = zeros(N, R)
    for i in 1:N
        u = ((0:R-1) .+ rand(R)) ./ R
        draws[i, :] .= quantile.(Normal(), u)
        shuffle!(draws[i, :])
    end
    return draws
end
