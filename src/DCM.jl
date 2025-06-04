"""
DCM.jl — A package for estimating Discrete Choice Models in Julia.

Inspired by Biogeme, DCM allows you to define symbolic utility functions
and estimate models like Multinomial Logit.
"""
module DCM

using Optim

include("Expressions.jl")
include("Models.jl")
include("Methods.jl")

end
