module TuringBenchmarking

using BenchmarkTools

using LogDensityProblems
using LogDensityProblemsAD

using Turing
using Turing.Essential: ForwardDiffAD, TrackerAD, ReverseDiffAD, ZygoteAD, CHUNKSIZE

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
    adbackends = DEFAULT_ADBACKENDS, run_once = true, save_grads = false
)
    suite = BenchmarkGroup()
    suite["not_linked"] = BenchmarkGroup()
    suite["linked"] = BenchmarkGroup()

    grads = Dict(:not_linked => Dict(), :linked => Dict())

    vi_orig = DynamicPPL.VarInfo(model)
    spl = DynamicPPL.SampleFromPrior()

    for adbackend in adbackends
        vi = DynamicPPL.VarInfo(vi_orig, spl, vi_orig[spl])
        f = LogDensityProblemsAD.ADgradient(
            adbackend,
            Turing.LogDensityFunction(vi, model, spl, DynamicPPL.DefaultContext())
        )
        θ = vi[spl]

        if run_once
            ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f, θ)

            if save_grads
                grads[:not_linked][adbackend] = (ℓ, ∇ℓ)
            end
        end
        suite["not_linked"]["$(adbackend)"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)

        # Need a separate `VarInfo` for the linked version since otherwise we risk the
        # `vi` from above being mutated.
        vi_linked = deepcopy(vi)
        DynamicPPL.link!(vi_linked, spl)
        f_linked = LogDensityProblemsAD.ADgradient(
            adbackend,
            Turing.LogDensityFunction(vi_linked, model, spl, DynamicPPL.DefaultContext())
        )
        θ_linked = vi_linked[spl]
        if run_once
            ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f_linked, θ_linked)

            if save_grads
                grads[:linked][adbackend] = (ℓ, ∇ℓ)
            end
        end
        suite["linked"]["$(adbackend)"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f_linked, $θ_linked)
    end

    # Also benchmark just standard model evaluation because why not.
    suite["not_linked"]["evaluation"] = @benchmarkable $(DynamicPPL.evaluate!!)($model, $vi_orig, $(DynamicPPL.DefaultContext()))
    DynamicPPL.link!(vi_orig, spl)
    suite["linked"]["evaluation"] = @benchmarkable $(DynamicPPL.evaluate!!)($model, $vi_orig, $(DynamicPPL.DefaultContext()))

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
function make_stan_suite(model::DynamicPPL.Model; kwargs...)
    error("`make_stan_suite` is not implemented. Try to load BridgeStan.jl to trigger definition of this function.")
end

# This symbol is only defined on Julia versions that support extensions
@static if !isdefined(Base, :get_extension)
    using Requires
    function __init__()
        @require BridgeStan = "c88b6f0a-829e-4b0b-94b7-f06ab5908f5a" include("../ext/TuringBenchmarkingBridgeStanExt.jl")
    end
end

end # module TuringBenchmarking
