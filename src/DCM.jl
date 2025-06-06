__precompile__()

"""
DCM.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia.

Features:
- Symbolic utilities using Parameters and Variables
- Support for Logit models with availability conditions
- Compatible with DataFrames
- Optim.jl-based estimation using automatic differentiation

Inspired by Biogeme, designed for Julia.
"""

module DCM

using Optim, DataFrames, ForwardDiff, LinearAlgebra, Distributions, Printf

include("Expressions.jl")
include("Models.jl")
include("Methods.jl")

export LogitModel, Parameter, Variable, estimate, summarize_results

end
