using TuringBenchmarking
using Documenter

DocMeta.setdocmeta!(TuringBenchmarking, :DocTestSetup, :(using TuringBenchmarking); recursive=true)

makedocs(;
    modules=[TuringBenchmarking],
    authors="Tor Erlend Fjelde <tor.erlend95@gmail.com> and contributors",
    repo="https://github.com/torfjelde/TuringBenchmarking.jl/blob/{commit}{path}#{line}",
    sitename="TuringBenchmarking.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://torfjelde.github.io/TuringBenchmarking.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/TuringBenchmarking.jl",
    devbranch="main",
)
