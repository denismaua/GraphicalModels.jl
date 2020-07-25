using GraphicalModels
using Documenter

makedocs(;
    modules=[GraphicalModels],
    authors="Denis Maua <denis.maua@gmail.com>",
    repo="https://github.com/denismaua/GraphicalModels.jl/blob/{commit}{path}#L{line}",
    sitename="GraphicalModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://denismaua.github.io/GraphicalModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/denismaua/GraphicalModels.jl",
)
