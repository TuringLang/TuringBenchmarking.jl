module TuringBenchmarking

using BenchmarkTools

using LogDensityProblems
using LogDensityProblemsAD

using Turing
using Turing.Essential: ForwardDiffAD, TrackerAD, ReverseDiffAD, ZygoteAD
using DynamicPPL: DynamicPPL

using ReverseDiff: ReverseDiff
using Zygote: Zygote

if !isdefined(Base, :get_extension)
    using Requires
end

export benchmark_model, make_turing_suite, @tagged

# Don't include `TrackerAD` because it's never going to win.
const DEFAULT_ADBACKENDS = [
    ForwardDiffAD{40}(),    # chunksize=40
    ForwardDiffAD{100}(),   # chunksize=100
    ZygoteAD(),
    ReverseDiffAD{false}(), # rdcache=false
    ReverseDiffAD{true}()   # rdcache=false
]

backend_label(::ForwardDiffAD) = "ForwardDiff"
backend_label(::ReverseDiffAD) = "ReverseDiff"
backend_label(::ZygoteAD) = "Zygote"
backend_label(::TrackerAD) = "Tracker"

const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ForwardDiffAD{40}(),
    :reversediff => ReverseDiffAD{false}(),
    :reversediff_compiled => ReverseDiffAD{true}(),
    :zygote => ZygoteAD(),
    :tracker => TrackerAD(),
)

to_backend(x) = error("Unknown backend: $x")
to_backend(x::Turing.Essential.ADBackend) = x
function to_backend(x::Union{AbstractString,Symbol})
    k = Symbol(lowercase(string(x)))
    haskey(SYMBOL_TO_BACKEND, k) || error("Unknown backend: $x")
    return SYMBOL_TO_BACKEND[k]
end

"""
    benchmark_model(model::Turing.Model; suite_kwargs..., kwargs...)

Create and run a benchmark suite for `model`.

The benchmarking suite will be created using [`make_turing_suite`](@ref).
See [`make_turing_suite`](@ref) for the available keyword arguments and more information.

# Keyword arguments
- `suite_kwargs`: Keyword arguments passed to [`make_turing_suite`](@ref).
- `kwargs`: Keyword arguments passed to `BenchmarkTools.run`.
"""
function benchmark_model(
    model::DynamicPPL.Model;
    adbackends = DEFAULT_ADBACKENDS,
    run_once::Bool = true,
    check_grads::Bool = false,
    varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(model),
    sampler::Union{AbstractMCMC.AbstractSampler,Nothing} = nothing,
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext(),
    verbose=true,
    kwargs...
)
    suite = make_turing_suite(
        model;
        adbackends,
        run_once,
        check_grads,
        varinfo,
        sampler,
        context,
        kwargs...
    )
    return run(suite; verbose=verbose, kwargs...)
end

"""
    make_turing_suite(model::Turing.Model; kwargs...)

Create default benchmark suite for `model`.

# Keyword arguments
- `adbackends`: a collection of adbackends to use, specified either as a
  `Turing.Essential.ADBackend` or using a `Symbol`. Defaults to `$(DEFAULT_ADBACKENDS)`.
- `run_once=true`: if `true`, the body of each benchmark will be run once to avoid
  compilation to be included in the timings (this may occur if compilation runs
  longer than the allowed time limit).
- `check_grads=false`: if `true` and `run_once` is `true`, the gradients from the initial
  execution will be compared against each other to ensure that they are consistent.
- `varinfo`: the `VarInfo` to use. Defaults to `DynamicPPL.VarInfo(model)`.
- `sampler`: the `Sampler` to use. Defaults to `nothing` (i.e. no sampler).
- `context`: the `Context` to use. Defaults to `DynamicPPL.DefaultContext()`.

# Notes
- A separate "parameter" instance (`DynamicPPL.VarInfo`) will be created for _each test_.
  Hence if you have a particularly large model, you might want to only pass one `adbackend`
  at the time.
"""
function make_turing_suite(
    model::DynamicPPL.Model;
    adbackends = DEFAULT_ADBACKENDS,
    run_once::Bool = true,
    check_grads::Bool = false,
    varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(model),
    sampler::Union{AbstractMCMC.AbstractSampler,Nothing} = nothing,
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    grads = Dict(:standard => Dict(), :linked => Dict())

    suite = BenchmarkGroup()
    suite_evaluation = BenchmarkGroup()
    suite_gradient = BenchmarkGroup()
    suite["evaluation"] = suite_evaluation
    suite["gradient"] = suite_gradient

    indexer = sampler === nothing ? Colon() : sampler
    if sampler !== nothing
        context = DynamicPPL.SamplingContext(sampler, context)
    end

    for adbackend in map(to_backend, adbackends)
        suite_backend = BenchmarkGroup([backend_label(adbackend)])
        suite_gradient["$(adbackend)"] = suite_backend

        suite_backend["standard"] = BenchmarkGroup()
        suite_backend["linked"] = BenchmarkGroup()

        varinfo_current = DynamicPPL.unflatten(varinfo, context, varinfo[indexer])
        f = LogDensityProblemsAD.ADgradient(
            adbackend,
            DynamicPPL.LogDensityFunction(
                varinfo_current, model, context
            )
        )
        θ = varinfo_current[indexer]

        try
            if run_once
                ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f, θ)

                if check_grads
                    grads[:standard][adbackend] = (ℓ, ∇ℓ)
                end
            end
            suite_backend["standard"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)
        catch e
            @warn "Gradient computation (without linking) failed for $(adbackend): $(e)"
        end

        # Need a separate `VarInfo` for the linked version since otherwise we risk the
        # `varinfo` from above being mutated.
        varinfo_linked = if sampler === nothing
            DynamicPPL.link!!(deepcopy(varinfo_current), model)
        else
            DynamicPPL.link!!(deepcopy(varinfo_current), sampler, model)
        end
        f_linked = LogDensityProblemsAD.ADgradient(
            adbackend,
            DynamicPPL.LogDensityFunction(varinfo_linked, model, context)
        )
        θ_linked = varinfo_linked[indexer]

        try
            if run_once
                ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f_linked, θ_linked)

                if check_grads
                    grads[:linked][adbackend] = (ℓ, ∇ℓ)
                end
            end
            suite_backend["linked"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f_linked, $θ_linked)
        catch e
            @warn "Gradient computation (with linking) failed for $(adbackend): $(e)"
        end
    end

    # Also benchmark just standard model evaluation because why not.
    suite_evaluation["standard"] = @benchmarkable $(DynamicPPL.evaluate!!)(
        $model, $varinfo, $context
    )
    varinfo_linked = if sampler === nothing
        DynamicPPL.link!!(deepcopy(varinfo), model)
    else
        DynamicPPL.link!!(deepcopy(varinfo), sampler, model)
    end
    suite_evaluation["linked"] = @benchmarkable $(DynamicPPL.evaluate!!)(
        $model, $varinfo_linked, $context
    )

    if check_grads
        # Let's check that the gradients are roughly the same for all backends.
        (val_first, grad_first) = first(values(grads[:standard]))
        for (backend, (val, grad)) in grads[:standard]
            if !(val ≈ val_first)
                @warn "Gradient check failed for $(backend): log-density values differ"
            end

            if !(grad ≈ grad_first)
                @warn "Gradient check failed for $(backend): gradients differ"
            end
        end
    end
    
    return suite
end

"""
    extract_stan_data(model::DynamicPPL.Model)

Return the data in `model` in a format consumable by the corresponding Stan model.

The Stan model requires the return data to be either
1. A JSON string representing a dictionary with the data.
2. A path to a data file ending in `.json`.
"""
function extract_stan_data end

"""
    stan_model_string(model::DynamicPPL.Model)

Return a string defining the Stan model corresponding to `model`.
"""
function stan_model_string end

"""
    make_stan_suite(model::Turing.Model; kwargs...)

Create default benchmark suite for the Stan model corresponding to `model`.
"""
function make_stan_suite end

# This symbol is only defined on Julia versions that support extensions
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require BridgeStan = "c88b6f0a-829e-4b0b-94b7-f06ab5908f5a" include("../ext/TuringBenchmarkingBridgeStanExt.jl")
    end
end

end # module TuringBenchmarking
