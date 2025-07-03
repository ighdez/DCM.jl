# DCM.jl — A symbolic, extensible package for estimating Discrete Choice Models in Julia

**DCM.jl** is a Julia package designed for flexible and symbolic specification of Discrete Choice Models (DCM). Inspired by **Biogeme** and **Apollo**, it uses symbolic algebra, modular architecture, and Julia's type system to allow rapid prototyping and estimation of models like Logit and Mixed Logit.

---

## 🚀 Key Features

- **Symbolic utility expressions** using `Parameter`, `Variable`, and algebraic operators.
- **Multinomial Logit and Mixed Logit** estimation with availability conditions.
- **Integration with `DataFrames.jl`** for data handling.
- **Estimation via `Optim.jl`**, supporting automatic or analytic gradients.
- **Prediction tools** for probabilities and most likely alternatives.
- **Extensibility** for new models (e.g., Latent Class, Nested Logit).

---

## ✨ Quick Example: Estimating a Logit Model

```julia
using DCM, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

asc_car = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)
β_cost = Parameter(:β_cost, value=0.0)

V_car = asc_car + β_time * Variable(:time_car) + β_cost * Variable(:cost_car)
V_bus = β_time * Variable(:time_bus) + β_cost * Variable(:cost_bus)

model = LogitModel([V_car, V_bus]; data=df, availability=[trues(nrow(df)), trues(nrow(df))])
results = estimate(model, :choice)
probs = predict(model, results)
```

---

## 🌀 Example: Estimating a Mixed Logit Model

```julia
using DCM, CSV, DataFrames

df = CSV.read("my_data.csv", DataFrame)

asc_car = Parameter(:asc_car, value=0.0)
β_time = Parameter(:β_time, value=0.0)
σ_time = Parameter(:σ_time, value=1.0)

draw = Draw(:time_rnd)
V_car = asc_car + (β_time + σ_time * draw) * Variable(:time_car)
V_bus = (β_time + σ_time * draw) * Variable(:time_bus)

model = MixedLogitModel([V_car, V_bus];
    data=df,
    idvar=:id,
    R=500,
    draw_scheme=:mlhs,
    availability=[trues(nrow(df)), trues(nrow(df))])

results = estimate(model, :choice)
P = predict(model, results)
```

---

## 📦 Installation

```julia
] dev https://github.com/ighdez/DCM.jl
```

---

## 🧱 Architecture Overview

- **Expressions**: `Parameter`, `Variable`, `Draw`, and algebraic combinators
- **Models**: `LogitModel`, `MixedLogitModel` (more to come!)
- **Estimation**: `estimate(model, choices)` using `Optim.jl`
- **Prediction**: `predict(model, results)` returns probabilities

---

## 📚 Documentation

All public functions are documented via Julia docstrings. Use `?estimate` in the REPL to see inline help. Full documentation will be published using Documenter.jl soon.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

- [Biogeme](https://biogeme.epfl.ch/)
- [Apollo](https://www.apollochoicemodelling.com/)
- [Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html)

Built with ♥ and `ChatGPT` pair-programming.
