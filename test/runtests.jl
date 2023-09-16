using TuringBenchmarking
using BenchmarkTools
using Turing
using Test

using Zygote: Zygote
using ReverseDiff: ReverseDiff

# Just make things run a bit faster.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.samples = 2

# These should be ordered (ascendingly) by runtime.
ADBACKENDS = TuringBenchmarking.DEFAULT_ADBACKENDS

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
        y, i, p, _, _ = sim(5, 3)

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
            DynamicPPL.SimpleVarInfo(model),
        ]
            suite = TuringBenchmarking.make_turing_suite(
                model;
                adbackends=ADBACKENDS,
                varinfo=varinfo,
                check_grads=true,
            )
            results = run(suite, verbose=true)

            @testset "$adbackend" for (i, adbackend) in enumerate(ADBACKENDS)
                adbackend_string = "$(adbackend)"
                results_backend = results[@tagged adbackend_string]
                # Each AD backend should have two results.
                @test length(leaves(results_backend)) == 2
                # It should be under the "gradient" section.
                @test haskey(results_backend, "gradient")
                # It should have one tagged "linked" and one "standard"
                @test length(leaves(results_backend[@tagged "linked"])) == 1
                @test length(leaves(results_backend[@tagged "standard"])) == 1
            end
        end

        @testset "Specify AD backends using symbols" begin
            varinfo = DynamicPPL.VarInfo(model)
            suite = TuringBenchmarking.make_turing_suite(
                model;
                adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote],
                varinfo=varinfo,
            )
            results = run(suite, verbose=true)

            @testset "$adbackend" for (i, adbackend) in enumerate(ADBACKENDS)
                adbackend_string = "$(adbackend)"
                results_backend = results[@tagged adbackend_string]
                # Each AD backend should have two results.
                @test length(leaves(results_backend)) == 2
                # It should be under the "gradient" section.
                @test haskey(results_backend, "gradient")
                # It should have one tagged "linked" and one "standard"
                @test length(leaves(results_backend[@tagged "linked"])) == 1
                @test length(leaves(results_backend[@tagged "standard"])) == 1
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
            results = run(suite, verbose=true)

            @testset "$adbackend" for (i, adbackend) in enumerate(ADBACKENDS)
                adbackend_string = "$(adbackend)"
                results_backend = results[@tagged adbackend_string]
                if adbackend isa TuringBenchmarking.ZygoteAD
                    # Zygote.jl should fail, i.e. return an empty suite.
                    @test length(leaves(results_backend)) == 0
                else
                    # Each AD backend should have two results.
                    @test length(leaves(results_backend)) == 2
                    # It should be under the "gradient" section.
                    @test haskey(results_backend, "gradient")
                    # It should have one tagged "linked" and one "standard"
                    @test length(leaves(results_backend[@tagged "linked"])) == 1
                    @test length(leaves(results_backend[@tagged "standard"])) == 1
                end
            end
        end
    end
end

