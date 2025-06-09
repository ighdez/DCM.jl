using CSV, DataFrames, Statistics
using DCM

# Load Swiss route choice dataset
df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)

# Define parameters for lognormal random coefficients
# b_tt = -exp(mu_tt + sigma_tt * draw_tt)
mu_tt     = Parameter(:mu_tt, value=0.0)
sigma_tt  = Parameter(:sigma_tt, value=1.0)

mu_tc     = Parameter(:mu_tc, value=0.0)
sigma_tc  = Parameter(:sigma_tc, value=1.0)

# Define variables
tt = Draw(:tt)  # draw for travel time
tc = Draw(:tc)  # draw for travel cost

# Define random parameters
b_tt = -exp(mu_tt + sigma_tt * tt)
b_tc = - exp(mu_tc + sigma_tc * tc)

# Define utility functions (alternatives 1, 2, 3)
V1 = b_tt * Variable(:tt1) + b_tc * Variable(:tc1)
V2 = b_tt * Variable(:tt2) + b_tc * Variable(:tc2)
V3 = b_tt * Variable(:tt3) + b_tc * Variable(:tc3)

all_params = collect_parameters.([V1, V2, V3]) |> Iterators.flatten |> collect
parameters = Dict(p.name => p.value for p in all_params)
for p in all_params
    println("Found parameter: ", p.name, " = ", p.value)
end
# exit()

for key in [:mu_tt, :sigma_tt, :mu_tc, :sigma_tc]
    @assert haskey(parameters, key) "Missing parameter: $key"
end
# Define utility functions (alternatives 1, 2, 3)
# V1 = -exp(mu_tt + sigma_tt * tt) * Variable(:tt1) - exp(mu_tc + sigma_tc * tc) * Variable(:tc1)
# V2 = -exp(mu_tt + sigma_tt * tt) * Variable(:tt2) - exp(mu_tc + sigma_tc * tc) * Variable(:tc2)
# V3 = -exp(mu_tt + sigma_tt * tt) * Variable(:tt3) - exp(mu_tc + sigma_tc * tc) * Variable(:tc3)

utilities = [V1, V2, V3]

# Availability (assumed all available for simplicity)
availability = [trues(nrow(df)) for _ in 1:3]

# Build and estimate the Mixed Logit model
model = MixedLogitModel(utilities; data=df, parameters=Dict(), availability=availability, R=200, draw_scheme=:mlhs)
results = estimate(model, df.choice)

# Output results
summarize_results(results)

# Predict probabilities
probs = predict(model)
println("\nAverage predicted choice probabilities:")
println(mean(probs, dims=1))
