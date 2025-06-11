using CSV, DataFrames, Statistics
using DCM

# Load Swiss route choice dataset
df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)

# Define parameters for lognormal random coefficients
mu_tt     = Parameter(:mu_tt, value=-1.9850)
sigma_tt  = Parameter(:sigma_tt, value=0.4740)

mu_tc     = Parameter(:mu_tc, value=-1.0228)
sigma_tc  = Parameter(:sigma_tc, value=1.0032)

mu_hw     = Parameter(:mu_hw, value=-2.9325)
sigma_hw  = Parameter(:sigma_hw, value=0.8046)

mu_ch     = Parameter(:mu_ch, value=0.6278)
sigma_ch  = Parameter(:sigma_ch, value=0.8456)

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
model = MixedLogitModel(utilities; data=df, id=df.ID, availability=availability, R=500, draw_scheme=:mlhs)
results = estimate(model, df.choice)

# Output results
summarize_results(results)

# Predict probabilities
probs = predict(model)
println("\nAverage predicted choice probabilities:")
println(mean(probs, dims=1))
