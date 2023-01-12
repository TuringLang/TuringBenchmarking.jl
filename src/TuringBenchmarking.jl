module TuringBenchmarking

using BenchmarkTools

using LogDensityProblems
using LogDensityProblemsAD

using Turing
using Turing.Essential: ForwardDiffAD, TrackerAD, ReverseDiffAD, ZygoteAD, CHUNKSIZE

using PyCall

const pystan = PyNULL()

function init_pystan()
    copy!(pystan, pyimport_conda("pystan", "pystan", "conda-forge"))
end

function __init__()
    try
        init_pystan()
    catch e
        @warn "Failed to import PyStan; related functionality will not work. Try manually calling `TuringBenchmarking.init_pystan()` for more info."
    end
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
    make_turing_suite(model; kwargs...)

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

Return a `Dict` which can be consumed by the corresponding Stan model.
"""
function extract_stan_data end

"""
    stan_model_string(model::DynamicPPL.Model)

Return a string defining the Stan model corresponding to `model`.
"""
function stan_model_string end


const STAN_DEFAULT_COMPILE_FLAGS = [
    "-ftemplate-depth-256",
    "-O3",
    "-mtune=native",
    "-march=native",
    "-pipe",
    "-fno-trapping-math",
    "-funroll-loops",
    "-funswitch-loops"
]

function make_stan_suite(
    model::DynamicPPL.Model;
    θ = nothing,
    model_string = nothing,
    initial_iters = 10,
    num_steps = 1,
    step_size = 1e-3,
    extra_compile_args = STAN_DEFAULT_COMPILE_FLAGS,
)
    if isnothing(model_string)
        model_string = stan_model_string(model)
    end

    sm = TuringBenchmarking.pystan.StanModel(
        model_code = model_string,
        # NOTE: Extra compile args doesn't seem to make any difference performnace-wise.
        extra_compile_args = extra_compile_args
    )

    # Convert the data/observations into something consumable by the Stan model.
    data = extract_stan_data(model)

    # Run a tiny bit of sampling because we need the resulting object to compute gradients.
    f = sm.sampling(
        data = data,
        iter = initial_iters,
        chains = 1,
        warmup = 0,
        algorithm = "HMC",
        control = Dict(
            "adapt_engaged" => false,
            # HMC
            "int_time" => num_steps * step_size,
            "metric" => "diag_e",
            "stepsize" => step_size,
            "stepsize_jitter" => 0,
        )
    )

    # Initialize from chain if parameters have not been provided.
    if isnothing(θ)
        θ = f.unconstrain_pars(f.get_last_position()[1])
    end

    # Create suite.
    stan_suite = BenchmarkGroup()
    stan_suite["evaluation"] = @benchmarkable $(f.log_prob)($θ)
    stan_suite["gradient"] = @benchmarkable $(f.grad_log_prob)($θ)

    return stan_suite
end

end # module TuringBenchmarking
