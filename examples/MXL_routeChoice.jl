using CSV, DataFrames, Statistics
using DCM

# Load Swiss route choice dataset
df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)

# Define parameters for lognormal random coefficients
mu_tt     = Parameter(:mu_tt, value=-2)
sigma_tt  = Parameter(:sigma_tt, value=0.001)

mu_tc     = Parameter(:mu_tc, value=-2)
sigma_tc  = Parameter(:sigma_tc, value=0.001)

mu_hw     = Parameter(:mu_hw, value=-2)
sigma_hw  = Parameter(:sigma_hw, value=0.001)

mu_ch     = Parameter(:mu_ch, value=-2)
sigma_ch  = Parameter(:sigma_ch, value=0.001)

# Define variables
tt = Draw(:tt)  # draw for travel time
tc = Draw(:tc)  # draw for travel cost
hw = Draw(:hw)  # draw for travel time
ch = Draw(:ch)  # draw for travel cost

# Define random parameters
b_tt = -exp(mu_tt + sigma_tt * tt)
b_tc = -exp(mu_tc + sigma_tc * tc)
b_hw = -exp(mu_hw + sigma_hw * hw)
b_ch = -exp(mu_ch + sigma_ch * ch)

# Define utility functions (alternatives 1, 2)
V1 = b_tt * Variable(:tt1) + b_tc * Variable(:tc1) + b_hw * Variable(:hw1) + b_ch * Variable(:ch1)
V2 = b_tt * Variable(:tt2) + b_tc * Variable(:tc2) + b_hw * Variable(:hw2) + b_ch * Variable(:ch2)

utilities = [V1, V2]

# Availability (assumed all available for simplicity)
availability = [
    trues(nrow(df)),
    trues(nrow(df))
]

# Build and estimate the Mixed Logit model
model = MixedLogitModel(utilities; data=df, id=df.ID, availability=availability, R=500, draw_scheme=:uniform)
params = collect_parameters(model.utilities)
for p in params
    println("Name: ", p.name, ", Value: ", p.value)
end

# params = Dict(
#     :mu_tt    => -2.0,
#     :sigma_tt => 0.001,
#     :mu_tc    => -2.0,
#     :sigma_tc => 0.001,
#     :mu_hw    => -2.0,
#     :sigma_hw => 0.001,
#     :mu_ch    => -2.0,
#     :sigma_ch => 0.001
# )

# model = MixedLogitModel(utilities; data=df, id=df.ID, availability=availability, parameters = params, R=500, draw_scheme=:uniform)
# @show loglikelihood(model, df.choice)
# exit()
results = estimate(model, df.choice)

# Output results
summarize_results(results)

# Predict probabilities
probs = predict(model)
println("\nAverage predicted choice probabilities:")
println(mean(probs, dims=1))
