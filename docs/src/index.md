```@meta
CurrentModule = TuringBenchmarking
```

# TuringBenchmarking.jl

A useful package for benchmarking and checking [Turing.jl](https://github.com/TuringLang/Turing.jl) models.

## Example

```@repl
using TuringBenchmarking, Turing

@model function demo(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

model = demo([1.5, 2.0]);

benchmark_model(
    model;
    # Check correctness of computations
    check=true,
    # Automatic differentiation backends to check and benchmark
    adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote]
)
```

## API

```@index
```

```@autodocs
Modules = [TuringBenchmarking]
```
