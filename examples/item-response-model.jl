using TuringBenchmarking,
    BridgeStan,
    BenchmarkTools,
    Turing,
    Zygote,
    ReverseDiff,
    ForwardDiff,
    JSON

### Setup ###
function sim(I, P)
    yvec = Vector{Int}(undef, I * P)
    ivec = similar(yvec)
    pvec = similar(yvec)

    beta = rand(Normal(), I)
    theta = rand(Normal(), P)

    n = 0
    for i in 1:I, p in 1:P
        n += 1
        ivec[n] = i
        pvec[n] = p
        yvec[n] = rand(BernoulliLogit(theta[p] - beta[i]))
    end

    return yvec, ivec, pvec, theta, beta
end

P = 1000
y, i, p, _, _ = sim(20, P);

### Turing ###
# naive implementation
@model function irt_naive(y, i, p; I = maximum(i), P = maximum(p))
    theta ~ filldist(Normal(), P)
    beta ~ filldist(Normal(), I)

    for n in eachindex(y)
        y[n] ~ Bernoulli(logistic(theta[p[n]] - beta[i[n]]))
    end
end

# performant model
@model function irt(y, i, p; I = maximum(i), P = maximum(p))
    theta ~ filldist(Normal(), P)
    beta ~ filldist(Normal(), I)
    Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(theta[p] - beta[i]), y))

    return (; theta, beta)
end

# Instantiate
model = irt(y, i, p);

# Make the benchmark suite.
suite = TuringBenchmarking.make_turing_suite(
    model,
    adbackends = [TuringBenchmarking.ForwardDiffAD{40}(), TuringBenchmarking.ReverseDiffAD{true}()]
);

# Run suite!
@info "Turing.jl" run(suite)


### Stan ###
@info "Compiling Stan model..."

# Tell `TuringBenchmarking` how to convert `model` into data consumable by Stan.
function TuringBenchmarking.extract_stan_data(model::DynamicPPL.Model{typeof(irt)})
    args = Dict(zip(string.(keys(model.args)), values(model.args)))
    kwargs = Dict(zip(string.(keys(model.defaults)), values(model.defaults)))
    kwargs["N"] = kwargs["I"] * kwargs["P"]
    return JSON.json(merge(args, kwargs))
end

# Tell `TuringBenchmarking` about the corresponding Stan model.
TuringBenchmarking.stan_model_string(model::DynamicPPL.Model{typeof(irt)}) = """
data {
    int<lower=1> I;
    int<lower=1> P;
    int<lower=1> N;
    int<lower=1, upper=I> i[N];
    int<lower=1, upper=P> p[N];
    int<lower=0, upper=1> y[N];
}
parameters {
    vector[I] beta;
    vector[P] theta;
}
model {
    theta ~ std_normal();
    beta ~ std_normal();
    y ~ bernoulli_logit(theta[p] - beta[i]);
}
"""

# Construct benchmark suite.
stan_suite = TuringBenchmarking.make_stan_suite(model)
# Run suite!
@info "Stan" run(stan_suite)

