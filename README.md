# TuringBenchmarking.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://torfjelde.github.io/TuringBenchmarking.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://torfjelde.github.io/TuringBenchmarking.jl/dev/)
[![Build Status](https://github.com/torfjelde/TuringBenchmarking.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/torfjelde/TuringBenchmarking.jl/actions/workflows/CI.yml?query=branch%3Amain)

A quick and dirty way to compare different automatic-differentiation backends in Turing.jl.

A typical workflow will look something like

``` julia
using BenchmarkTools
using Turing
using TuringBenchmarking

# Define your model.
@model function your_model(...)
    # ...
end

# Create and run the benchmarking suite for Turing.jl.
turing_suite = make_turing_suite(model; kwargs...)
run(turing_suite)
```

Example output:

``` julia
# Running `examples/item-response-model.jl` on my laptop.
2-element BenchmarkTools.BenchmarkGroup:
  tags: []
  "linked" => 2-element BenchmarkTools.BenchmarkGroup:
          tags: []
          "evaluation" => Trial(1.132 ms)
          "Turing.Essential.ReverseDiffAD{true}()" => Trial(1.598 ms)
          "Turing.Essential.ForwardDiffAD{40, true}()" => Trial(184.703 ms)
  "not_linked" => 2-element BenchmarkTools.BenchmarkGroup:
          tags: []
          "evaluation" => Trial(1.131 ms)
          "Turing.Essential.ReverseDiffAD{true}()" => Trial(1.596 ms)
          "Turing.Essential.ForwardDiffAD{40, true}()" => Trial(182.864 ms)
```

`"linked"`/`"not_linked"` here refers to whether or not we're working in unconstrained space.

And if you want to compare the result to Stan:

``` julia
using BridgeStan, JSON

# Tell `TuringBenchmarking` how to convert `model` into Stan data & model.
function TuringBenchmaring.extract_stan_data(model::Turing.Model{typeof(your_model)})
    # In the case where the Turing.jl and Stan models are identical in what they expect we can just do:
    return JSON.json(Dict(zip(string.(keys(model.args)), values(model.args))))
end

TuringBenchmarking.stan_model_string(model::Turing.Model{typeof(your_model)}) = """
[HERE GOES YOUR STAN MODEL DEFINITION]
"""

# Create and run the benchmarking suite for Stan.
stan_suite = make_stan_suite(model; kwargs...)
run(stan_suite)
```

Example output:

``` julia
# Running `examples/item-response-model.jl` on my laptop.
2-element BenchmarkTools.BenchmarkGroup:
  tags: []
  "evaluation" => Trial(1.102 ms)
  "gradient" => Trial(1.256 ms)
```

Note that the benchmarks for Stan are only in unconstrained space.
