using CSV, DataFrames
using DCM

# Load dataset and filter RP observations only
df = CSV.read("../data/apollo_modeChoiceData.csv", DataFrame)
df = filter(:SP => x -> x == 1, df)

# Define variables and parameters
asc_bus = Parameter(:asc_bus, value=0)
asc_air = Parameter(:asc_air, value=0)
asc_rail = Parameter(:asc_rail, value=0)

β_time_car = Parameter(:β_time_car, value=0)
β_time_bus = Parameter(:β_time_bus, value=0)
β_time_air = Parameter(:β_time_air, value=0)
β_time_rail = Parameter(:β_time_rail, value=0)

β_access = Parameter(:β_access, value=0)
β_cost = Parameter(:β_cost, value=0)
β_wifi = Parameter(:β_wifi, value=0)
β_food = Parameter(:β_food, value=0)

utilities = [
    β_time_car * Variable(:time_car) + β_cost * Variable(:cost_car),
    asc_bus  + β_time_bus * Variable(:time_bus)   + β_access * Variable(:access_bus)  + β_cost * Variable(:cost_bus),
    asc_air  + β_time_air * Variable(:time_air)   + β_access * Variable(:access_air)  + β_cost * Variable(:cost_air) + β_wifi * (Variable(:service_air) == 2)  + β_food * (Variable(:service_air) == 3),
    asc_rail + β_time_rail * Variable(:time_rail) + β_access * Variable(:access_rail) + β_cost * Variable(:cost_rail) + β_wifi * (Variable(:service_rail) == 2 ) + β_food * (Variable(:service_rail) == 3)
]

# Define choice vector (1:car, 2:bus, 3:air, 4:rail)
choices = convert(Vector{Int}, df.choice)

# Load availability data
availability = [
    df.av_car .== 1,
    df.av_bus .== 1,
    df.av_air .== 1,
    df.av_rail .== 1
]

# Define parameter dictionary
params = Dict(
    :asc_bus => 0., :asc_air => 0., :asc_rail => 0.,
    :β_time_car => 0., :β_time_bus => 0., :β_time_air => 0., :β_time_rail => 0.,
    :β_access => 0., :β_cost => 0.,
    :β_wifi => 0., :β_food => 0.)

# Create model and estimate
model = LogitModel(utilities; data=df, parameters=params, availability=availability)
results = estimate(model, choices)

# Output results
println("Estimated Parameters:")
for (k, v) in results.parameters
    println("  ", k, " = ", round(v, digits=4))
end

println("Log-likelihood: ", round(results.loglikelihood, digits=4))
