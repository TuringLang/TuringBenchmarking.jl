using TuringBenchmarking
using BenchmarkTools
using Turing
using Test

using Zygote: Zygote
using ReverseDiff: ReverseDiff

@testset "TuringBenchmarking.jl" begin
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

    P = 10
    y, i, p, _, _ = sim(20, P)

    ### Turing ###
    # performant model
    @model function irt(y, i, p; I=maximum(i), P=maximum(p))
        theta ~ filldist(Normal(), P)
        beta ~ filldist(Normal(), I)
        Turing.@addlogprob! sum(logpdf.(BernoulliLogit.(theta[p] - beta[i]), y))

        return (; theta, beta)
    end

    # Instantiate
    model = irt(y, i, p)

    # Make the benchmark suite.
    # These should be ordered (ascendingly) by runtime.
    adbackends = [
        TuringBenchmarking.ForwardDiffAD{40}(),
        TuringBenchmarking.ReverseDiffAD{true}(),
        TuringBenchmarking.ReverseDiffAD{false}(),
        TuringBenchmarking.ZygoteAD(),
    ]
    suite = TuringBenchmarking.make_turing_suite(
        model,
        adbackends=adbackends
    )
    results = run(suite)

    # TODO: Is there a better way to test these?
    for (i, adbackend) in enumerate(adbackends)
        @test haskey(suite["not_linked"], "$(adbackend)")
        @test haskey(suite["linked"], "$(adbackend)")

        if i < length(adbackends)
            @test median(results["not_linked"]["$(adbackend)"]) ≤ median(results["not_linked"]["$(adbackends[i+1])"])
            @test median(results["linked"]["$(adbackend)"]) ≤ median(results["linked"]["$(adbackends[i+1])"])
        end
    end
end
