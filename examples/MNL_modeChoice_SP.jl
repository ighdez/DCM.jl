using CSV, DataFrames
using DCM

# Load dataset and filter RP observations only
df = CSV.read("../data/apollo_modeChoiceData.csv", DataFrame)
df = filter(:SP => x -> x == 1, df)

# Define variables and parameters
asc_car = Parameter(:asc_car, value=0, fixed=true)
asc_bus = Parameter(:asc_bus, value=0)
asc_air = Parameter(:asc_air, value=0)
asc_rail = Parameter(:asc_rail, value=0)

β_time_car = Parameter(:β_time_car, value=0)
β_time_bus = Parameter(:β_time_bus, value=0)
β_time_air = Parameter(:β_time_air, value=0)
β_time_rail = Parameter(:β_time_rail, value=0)

β_access = Parameter(:β_access, value=0)
β_cost = Parameter(:β_cost, value=0)
β_no_frills = Parameter(:β_no_frills, value=0, fixed=true)
β_wifi = Parameter(:β_wifi, value=0)
β_food = Parameter(:β_food, value=0)

V1 = asc_car  + β_time_car * Variable(:time_car) + β_cost * Variable(:cost_car)
V2 = asc_bus  + β_time_bus * Variable(:time_bus)   + β_access * Variable(:access_bus)  + β_cost * Variable(:cost_bus)
V3 = asc_air  + β_time_air * Variable(:time_air)   + β_access * Variable(:access_air)  + β_cost * Variable(:cost_air) + β_no_frills * (Variable(:service_air) == 1) + β_wifi * (Variable(:service_air) == 2)  + β_food * (Variable(:service_air) == 3)
V4 = asc_rail + β_time_rail * Variable(:time_rail) + β_access * Variable(:access_rail) + β_cost * Variable(:cost_rail) + β_no_frills * (Variable(:service_rail) == 1) + β_wifi * (Variable(:service_rail) == 2 ) + β_food * (Variable(:service_rail) == 3)

utilities = [V1,V2,V3,V4]

# Load availability data
availability = [
    df.av_car .== 1,
    df.av_bus .== 1,
    df.av_air .== 1,
    df.av_rail .== 1
]

# Create model and estimate
model = LogitModel(utilities; data=df, availability=availability)
results = estimate(model, df.choice)

# Output results
summarize_results(results)

# Predict
preds = predict(model,results)
println("\nAverage of Logit predictions")
println(mean(preds,dims=1))