var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TuringBenchmarking","category":"page"},{"location":"#TuringBenchmarking.jl","page":"Home","title":"TuringBenchmarking.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A useful package for benchmarking and checking Turing.jl models.","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using TuringBenchmarking, Turing\n\n@model function demo(x)\n    s ~ InverseGamma(2, 3)\n    m ~ Normal(0, sqrt(s))\n    for i in 1:length(x)\n        x[i] ~ Normal(m, sqrt(s))\n    end\nend\n\nmodel = demo([1.5, 2.0]);\n\nbenchmark_model(\n    model;\n    # Check correctness of computations\n    check=true,\n    # Automatic differentiation backends to check and benchmark\n    adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote]\n)","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TuringBenchmarking]","category":"page"},{"location":"#TuringBenchmarking.benchmark_model-Tuple{DynamicPPL.Model}","page":"Home","title":"TuringBenchmarking.benchmark_model","text":"benchmark_model(model::Turing.Model; suite_kwargs..., kwargs...)\n\nCreate and run a benchmark suite for model.\n\nThe benchmarking suite will be created using make_turing_suite. See make_turing_suite for the available keyword arguments and more information.\n\nKeyword arguments\n\nsuite_kwargs: Keyword arguments passed to make_turing_suite.\nkwargs: Keyword arguments passed to BenchmarkTools.run.\n\n\n\n\n\n","category":"method"},{"location":"#TuringBenchmarking.extract_stan_data","page":"Home","title":"TuringBenchmarking.extract_stan_data","text":"extract_stan_data(model::DynamicPPL.Model)\n\nReturn the data in model in a format consumable by the corresponding Stan model.\n\nThe Stan model requires the return data to be either\n\nA JSON string representing a dictionary with the data.\nA path to a data file ending in .json.\n\n\n\n\n\n","category":"function"},{"location":"#TuringBenchmarking.make_stan_suite","page":"Home","title":"TuringBenchmarking.make_stan_suite","text":"make_stan_suite(model::Turing.Model; kwargs...)\n\nCreate default benchmark suite for the Stan model corresponding to model.\n\n\n\n\n\n","category":"function"},{"location":"#TuringBenchmarking.make_turing_suite-Tuple{DynamicPPL.Model}","page":"Home","title":"TuringBenchmarking.make_turing_suite","text":"make_turing_suite(model::Turing.Model; kwargs...)\n\nCreate default benchmark suite for model.\n\nKeyword arguments\n\nadbackends: a collection of adbackends to use, specified either as a type from\n\nADTypes.jl or using a Symbol. Defaults to ADTypes.AbstractADType[ADTypes.AutoForwardDiff(chunksize=0), ADTypes.AutoReverseDiff(), ADTypes.AutoReverseDiff(compile=true), ADTypes.AutoZygote(), ADTypes.AutoMooncake{Nothing}(nothing)].\n\nrun_once=true: if true, the body of each benchmark will be run once to avoid compilation to be included in the timings (this may occur if compilation runs longer than the allowed time limit).\ncheck=false: if true, the log-density evaluations and the gradients will be compared against each other to ensure that they are consistent. Note that this will force run_once=true.\nerror_on_failed_check=false: if true, an error will be thrown if the check fails rather than just printing a warning, as is done by default.\nerror_on_failed_backend=false: if true, an error will be thrown if the evaluation of the log-density or the gradient fails for any of the backends rather than just printing a warning, as is done by default.\nvarinfo: the VarInfo to use. Defaults to DynamicPPL.VarInfo(model).\nsampler: the Sampler to use. Defaults to nothing (i.e. no sampler).\ncontext: the Context to use. Defaults to DynamicPPL.DefaultContext().\nθ: the parameters to use. Defaults to rand(Vector, model).\nθ_linked: the linked parameters to use. Defaults to randn(d) where d  is the length of the linked parameters..\natol: the absolute tolerance to use for comparisons.\nrtol: the relative tolerance to use for comparisons.\n\nNotes\n\nA separate \"parameter\" instance (DynamicPPL.VarInfo) will be created for each test. Hence if you have a particularly large model, you might want to only pass one adbackend at the time.\n\n\n\n\n\n","category":"method"},{"location":"#TuringBenchmarking.stan_model_string","page":"Home","title":"TuringBenchmarking.stan_model_string","text":"stan_model_string(model::DynamicPPL.Model)\n\nReturn a string defining the Stan model corresponding to model.\n\n\n\n\n\n","category":"function"}]
}
