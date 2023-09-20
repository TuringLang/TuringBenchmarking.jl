module TuringBenchmarking

using LinearAlgebra
using BenchmarkTools

using LogDensityProblems
using LogDensityProblemsAD

using PrettyTables: PrettyTables

using Turing
using Turing.Essential: ForwardDiffAD, TrackerAD, ReverseDiffAD, ZygoteAD
using DynamicPPL: DynamicPPL

using ReverseDiff: ReverseDiff
using Zygote: Zygote

if !isdefined(Base, :get_extension)
    using Requires
end

export benchmark_model, make_turing_suite, BenchmarkTools, @tagged

# Don't include `TrackerAD` because it's never going to win.
const DEFAULT_ADBACKENDS = [
    ForwardDiffAD{Turing.Essential.CHUNKSIZE[]}(), # chunksize=40
    ReverseDiffAD{false}(), # rdcache=false
    ReverseDiffAD{true}(),  # rdcache=false
    ZygoteAD(),
]

backend_label(::ForwardDiffAD) = "ForwardDiff"
backend_label(::ReverseDiffAD{false}) = "ReverseDiff"
backend_label(::ReverseDiffAD{true}) = "ReverseDiff[compiled]"
backend_label(::ZygoteAD) = "Zygote"
backend_label(::TrackerAD) = "Tracker"

const SYMBOL_TO_BACKEND = Dict(
    :forwarddiff => ForwardDiffAD{Turing.Essential.CHUNKSIZE[]}(),
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
    check::Bool = false,
    check_grads::Bool = check,
    error_on_failed_check::Bool = false,
    error_on_failed_backend::Bool = false,
    varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(model),
    sampler::Union{AbstractMCMC.AbstractSampler,Nothing} = nothing,
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext(),
    atol::Real = 1e-6,
    rtol::Real = 0,
    kwargs...
)
    suite = make_turing_suite(
        model;
        adbackends,
        run_once,
        check,
        check_grads,
        varinfo,
        sampler,
        context,
        atol,
        rtol,
    )
    return run(suite; kwargs...)
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
- `check=false`: if `true`, the log-density evaluations and the gradients
  will be compared against each other to ensure that they are consistent.
  Note that this will force `run_once=true`.
- `error_on_failed_check=false`: if `true`, an error will be thrown if the
  check fails rather than just printing a warning, as is done by default.
- `error_on_failed_backend=false`: if `true`, an error will be thrown if the
  evaluation of the log-density or the gradient fails for any of the backends
  rather than just printing a warning, as is done by default.
- `varinfo`: the `VarInfo` to use. Defaults to `DynamicPPL.VarInfo(model)`.
- `sampler`: the `Sampler` to use. Defaults to `nothing` (i.e. no sampler).
- `context`: the `Context` to use. Defaults to `DynamicPPL.DefaultContext()`.
- `atol`: the absolute tolerance to use for comparisons.
- `rtol`: the relative tolerance to use for comparisons.

# Notes
- A separate "parameter" instance (`DynamicPPL.VarInfo`) will be created for _each test_.
  Hence if you have a particularly large model, you might want to only pass one `adbackend`
  at the time.
"""
function make_turing_suite(
    model::DynamicPPL.Model;
    adbackends = DEFAULT_ADBACKENDS,
    run_once::Bool = true,
    check::Bool = false,
    check_grads::Bool = check,
    error_on_failed_check::Bool = false,
    error_on_failed_backend::Bool = false,
    varinfo::DynamicPPL.AbstractVarInfo = DynamicPPL.VarInfo(model),
    sampler::Union{AbstractMCMC.AbstractSampler,Nothing} = nothing,
    context::DynamicPPL.AbstractContext = DynamicPPL.DefaultContext(),
    atol::Real = 1e-6,
    rtol::Real = 0,
)
    if check != check_grads
        Base.depwarn(
            "The `check_grads` keyword argument is deprecated. Use `check` instead.",
            :make_turing_suite
        )
        check = check_grads
    end

    grads_and_vals = Dict(:standard => Dict(), :linked => Dict())
    adbackends = map(to_backend, adbackends)

    suite = BenchmarkGroup()
    suite_evaluation = BenchmarkGroup()
    suite_gradient = BenchmarkGroup()
    suite["evaluation"] = suite_evaluation
    suite["gradient"] = suite_gradient

    indexer = sampler === nothing ? Colon() : sampler
    if sampler !== nothing
        context = DynamicPPL.SamplingContext(sampler, context)
    end

    for adbackend in adbackends
        suite_backend = BenchmarkGroup([backend_label(adbackend)])
        suite_gradient["$(adbackend)"] = suite_backend

        suite_backend["standard"] = BenchmarkGroup()
        suite_backend["linked"] = BenchmarkGroup()

        varinfo_current = DynamicPPL.unflatten(varinfo, context, varinfo[indexer])
        f = LogDensityProblemsAD.ADgradient(
            adbackend,
            DynamicPPL.LogDensityFunction(varinfo_current, model, context)
        )
        θ = varinfo_current[indexer]

        try
            if run_once || check_grads
                ℓ, ∇ℓ = LogDensityProblems.logdensity_and_gradient(f, θ)
                @debug "$(backend_label(adbackend))" θ ℓ ∇ℓ

                if check_grads
                    grads_and_vals[:standard][adbackend] = (ℓ, ∇ℓ)
                end
            end
            suite_backend["standard"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f, $θ)
        catch e
            if error_on_failed_backend
                rethrow(e)
            else
                @warn "Gradient computation (without linking) failed for $(adbackend): $(e)"
            end
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
            if run_once || check_grads
                ℓ_linked, ∇ℓ_linked = LogDensityProblems.logdensity_and_gradient(f_linked, θ_linked)
                @debug "$(backend_label(adbackend)) [linked]" θ_linked ℓ_linked ∇ℓ_linked

                if check_grads
                    grads_and_vals[:linked][adbackend] = (ℓ_linked, ∇ℓ_linked)
                end
            end
            suite_backend["linked"] = @benchmarkable $(LogDensityProblems.logdensity_and_gradient)($f_linked, $θ_linked)
        catch e
            if error_on_failed_backend
                rethrow(e)
            else
                @warn "Gradient computation (with linking) failed for $(adbackend): $(e)"
            end
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
        success = true
        for type in [:standard, :linked]
            backends = collect(keys(grads_and_vals[type]))
            vals = map(first, values(grads_and_vals[type]))
            vals_dists = compute_distances(backends, vals)
            if !all(isapprox.(values(vals_dists), 0, atol=atol, rtol=rtol))
                @warn "There is disagreement in the log-density values!"
                show_distances(vals_dists; header=([titlecase(string(type)), "Log-density"], ["backend", "distance"]), atol=atol, rtol=rtol)
                success = false
            end
            grads = map(last, values(grads_and_vals[type]))
            grads_dists = compute_distances(backends, grads)
            if !all(isapprox.(values(grads_dists), 0, atol=atol, rtol=rtol))
                @warn "There is disagreement in the gradients!"
                show_distances(grads_dists, header=([titlecase(string(type)), "Gradient"], ["backend", "distance"]), atol=atol, rtol=rtol)
                success = false
            end
        end

        if !success && error_on_failed_check
            error("Consistency checks failed!")
        end
    end

    return suite
end

function compute_distances(backends, vals)
    T = eltype(first(vals))
    n = length(vals)
    dists = DynamicPPL.OrderedDict{String,T}()
    for (i, backend_i) in zip(1:n, backends)
        for (j, backend_j) in zip(i + 1:n, backends[i + 1:end])
            dists["$(backend_label(backend_i)) vs $(backend_label(backend_j))"] = norm(vals[i] - vals[j])
        end
    end

    return dists
end

function show_distances(dists::AbstractDict; header=["Backend", "Distance"], atol=1e-6, rtol=0)
    hl = PrettyTables.Highlighter(
        (data, i, j) -> !isapprox(data[i, 2], 0; atol=atol, rtol=rtol),
        PrettyTables.crayon"red bold"
    )
    PrettyTables.pretty_table(
        dists;
        header=header,
        highlighters=(hl,),
        formatters=PrettyTables.ft_printf("%.2f", [2])
    )
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
