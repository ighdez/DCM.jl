[![Build Status](https://github.com/ighdez/DCM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ighdez/DCM.jl/actions/workflows/CI.yml?query=branch%3Amain)


# DCM.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia

**DCM.jl** is a Julia package designed for flexible and symbolic specification of discrete choice models (DCM). Inspired by Biogeme and Apollo, this package leverages Julia's multiple dispatch and metaprogramming capabilities to deliver a modular, extensible framework.

---

## 🚀 Key Features

* **Symbolic expressions**: Define utility functions using symbolic `Parameter` and `Variable` objects.
<!-- * **Multinomial Logit estimation**: Native support for MNL models with availability conditions. -->
* **DataFrame integration**: Model data can be managed using `DataFrames.jl`.
* **Optim.jl backend**: Estimation routines are powered by `Optim.jl`, using automatic gradients.
* **Prediction tools**: Predict choice probabilities and most likely alternatives with minimal user intervention.
* **Multiple dispatch**: Extend to new models (e.g., Mixed Logit) via custom subtypes and model-specific methods.

---

## ✨ Example: Estimating a Logit Model

```julia
using DCM, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

# Define parameters
asc_car = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)
β_cost = Parameter(:β_cost, value=0.0)

# Define variables
V_car = asc_car + β_time * Variable(:time_car) + β_cost * Variable(:cost_car)
V_bus = β_time * Variable(:time_bus) + β_cost * Variable(:cost_bus)

# Create model
model = LogitModel([V_car, V_bus]; data=df, availability=[trues(nrow(df)), trues(nrow(df))])

# Estimate parameters
results = estimate(model, df.choice)

# Predict probabilities
probs = predict(model, results)               # Vector of probability vectors
```

---

## 🧱 Architecture Overview

DCM separates concerns into the following components:

### Expressions

* `Parameter(name; value, fixed)`: symbolic named parameter.
* `Variable(name)`: placeholder for a column in the data.
<!-- * Expression types: `DCMSum`, `DCMMult`, `DCMExp`, etc. -->

### Models

<!-- * Abstract type: `DiscreteChoiceModel` -->
* Concrete type: `LogitModel`
* Future extensions: `MixedLogitModel`, `RRMModel`, etc.

### Estimation

* `estimate(model::LogitModel, choices::Vector{Int})`

  * Returns a `NamedTuple` with parameter estimates, std errors, and convergence info.

### Prediction

* `predict(model::LogitModel, results)`: returns NxJ matrix of probabilities.

---

## 📚 Documentation in Source

All public methods are documented using Julia's docstring syntax (`""" ... """`). Run `?estimate` in the REPL for inline help.

---

## 📦 Installation

DCM.jl is currently in development. To install from source:

```julia
] dev https://github.com/yourusername/DCM.jl
```

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

DCM.jl is inspired by:

* [Biogeme](https://biogeme.epfl.ch/)
* [Apollo R package](https://www.apollochoicemodelling.com/)
* [Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html)

Development was supported in part by [ChatGPT](https://openai.com/chatgpt), through vibe-driven pair programming and documentation feedback. 💻✨

