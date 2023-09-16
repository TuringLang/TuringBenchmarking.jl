using TuringBenchmarking
using BenchmarkTools
using Turing
using Test

using Zygote: Zygote
using ReverseDiff: ReverseDiff

# These should be ordered (ascendingly) by runtime.
ADBACKENDS = [
    TuringBenchmarking.ForwardDiffAD{40}(),
    TuringBenchmarking.ReverseDiffAD{true}(),
    TuringBenchmarking.ReverseDiffAD{false}(),
    TuringBenchmarking.ZygoteAD(),
]

@testset "TuringBenchmarking.jl" begin
    @testset "Item-Response model" begin
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
        @testset "$(nameof(typeof(varinfo)))" for varinfo in [
            DynamicPPL.VarInfo(model),
            DynamicPPL.SimpleVarInfo{Float64}(model),
        ]
            suite = TuringBenchmarking.make_turing_suite(
                model;
                adbackends=ADBACKENDS,
                varinfo=varinfo
            )
            results = run(suite, verbose=true, evals=1, samples=2)

            # TODO: Is there a better way to test these?
            for (i, adbackend) in enumerate(ADBACKENDS)
                @test haskey(suite["not_linked"], "$(adbackend)")
                @test haskey(suite["linked"], "$(adbackend)")
            end
        end
    end

    @testset "Model with mutation" begin
        @model function demo_with_mutation(::Type{TV}=Vector{Float64}) where {TV}
            x = TV(undef, 2)
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end

        model = demo_with_mutation()

        # Make the benchmark suite.
        @testset "$(nameof(typeof(varinfo)))" for varinfo in [
            DynamicPPL.VarInfo(model),
            DynamicPPL.SimpleVarInfo(x=randn(2)),
            DynamicPPL.SimpleVarInfo(DynamicPPL.OrderedDict(@varname(x) => randn(2))),
        ]
            suite = TuringBenchmarking.make_turing_suite(
                model;
                adbackends=ADBACKENDS,
                varinfo=varinfo
            )
            results = run(suite, verbose=true, evals=1, samples=2)

            for (i, adbackend) in enumerate(ADBACKENDS)
                # Zygote.jl should fail.
                if adbackend isa TuringBenchmarking.ZygoteAD
                    @test !haskey(suite["not_linked"], "$(adbackend)")
                    @test !haskey(suite["linked"], "$(adbackend)")
                else
                    @test haskey(suite["not_linked"], "$(adbackend)")
                    @test haskey(suite["linked"], "$(adbackend)")
                end
            end
        end
    end
end

