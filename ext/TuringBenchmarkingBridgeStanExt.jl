module TuringBenchmarkingBridgeStanExt

if isdefined(Base, :get_extension)
    using TuringBenchmarking: TuringBenchmarking, BenchmarkTools
    using BridgeStan: BridgeStan
else
    using ..TuringBenchmarking: TuringBenchmarking, BenchmarkTools
    using ..BridgeStan: BridgeStan
end

function stan_model_to_file_maybe(x::AbstractString)
    endswith(x, ".stan") && return x

    # Write to a temporary file.
    tmpfile = tempname() * ".stan"
    open(tmpfile, "w") do io
        write(io, x)
    end

    return tmpfile
end

function TuringBenchmarking.make_stan_suite(
    model::TuringBenchmarking.DynamicPPL.Model;
    θ = nothing,
    model_string = nothing,
    stanc_args = String[],
    make_args = String[]
)

    if isnothing(model_string)
        model_string = TuringBenchmarking.stan_model_string(model)
    end

    # Convert the data/observations into something consumable by the Stan model.
    data = TuringBenchmarking.extract_stan_data(model)

    stan_model = BridgeStan.StanModel(
        stan_file = stan_model_to_file_maybe(model_string),
        data = data,
        stanc_args = stanc_args,
        make_args = make_args,
    )

    # Initialize from chain if parameters have not been provided.
    ϕ = if isnothing(θ)
        rand(BridgeStan.param_num(stan_model))
    else
        BridgeStan.param_unconstrain(stan_model, θ)
    end

    # Create suite.
    stan_suite = BenchmarkTools.BenchmarkGroup()
    stan_suite["evaluation"] = BenchmarkTools.@benchmarkable $(BridgeStan.log_density)(
        $stan_model, $ϕ
    )
    stan_suite["gradient"] = BenchmarkTools.@benchmarkable $(BridgeStan.log_density_gradient)(
        $stan_model, $ϕ
    )

    return stan_suite
end

end


