module TuringBenchmarking

using BenchmarkTools

using LogDensityProblems
using LogDensityProblemsAD

using Turing
using Turing.Essential: ForwardDiffAD, TrackerAD, ReverseDiffAD, ZygoteAD, CHUNKSIZE

if !isdefined(Base, :get_extension)
    using Requires
end

# Don't include `TrackerAD` because it's never going to win.
const DEFAULT_ADBACKENDS = [
    ForwardDiffAD{40}(),    # chunksize=40
    ForwardDiffAD{100}(),   # chunksize=100
    ZygoteAD(),
    ReverseDiffAD{false}(), # rdcache=false
    ReverseDiffAD{true}()   # rdcache=false
]

"""
    make_turing_suite(model::Turing.Model; kwargs...)

Create default benchmark suite for `model`.

# Keyword arguments
- `adbackends`: a collection of adbackends to use. Defaults to `$(DEFAULT_ADBACKENDS)`.
- `run_once=true`: if `true`, the body of each benchmark will be run once to avoid
  compilation to be included in the timings (this may occur if compilation runs
  longer than the allowed time limit).
- `save_grads=false`: if `true` and `run_once` is `true`, the gradients from the initial
  execution will be saved and returned as the second return-value. This is useful if you
  want to check correctness of the gradients for different backends.

# Notes
- A separate "parameter" instance (`DynamicPPL.VarInfo`) will be created for _each test_.
  Hence if you have a particularly large model, you might want to only pass one `adbackend`
  at the time.
"""
function make_turing_suite(
    model::DynamicPPL.Model;
    adbackends = DEFAULT_ADBACKENDS,
    run_once::Bool = true,
    save_grads::Bool = false,
    varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(model),
    sampler::Union{AbstractMCMC.AbstractSampler,Nothing} = nothing,
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    suite = BenchmarkGroup()
    suite["not_linked"] = BenchmarkGroup()
    suite["linked"] = BenchmarkGroup()

    grads = Dict(:not_linked => Dict(), :linked => Dict())

    indexer = sampler === nothing ? Colon() : sampler
    if sampler !== nothing
        context = DynamicPPL.SamplingContext(sampler, context)
    end

    for adbackend in adbackends
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

                if save_grads
                    grads[:not_linked][adbackend] = (ℓ, ∇ℓ)
                end
            end
            suite["not_linked"]["$(adbackend)"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)
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
            Turing.LogDensityFunction(varinfo_linked, model, context)
        )
        θ_linked = varinfo_linked[indexer]

        try
            if run_once
                ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f_linked, θ_linked)

                if save_grads
                    grads[:linked][adbackend] = (ℓ, ∇ℓ)
                end
            end
            suite["linked"]["$(adbackend)"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f_linked, $θ_linked)
        catch e
            @warn "Gradient computation (with linking) failed for $(adbackend): $(e)"
        end
    end

    # Also benchmark just standard model evaluation because why not.
    suite["not_linked"]["evaluation"] = @benchmarkable $(DynamicPPL.evaluate!!)($model, $varinfo, $context)
    varinfo_linked = if sampler === nothing
        DynamicPPL.link!!(deepcopy(varinfo), model)
    else
        DynamicPPL.link!!(deepcopy(varinfo), sampler, model)
    end
    suite["linked"]["evaluation"] = @benchmarkable $(DynamicPPL.evaluate!!)($model, $varinfo_linked, $context)

    return save_grads ? (suite, grads) : suite
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
