using RegisterQD
using Documenter

makedocs(;
    modules = [RegisterQD],
    authors = "HolyLab",
    sitename = "RegisterQD.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://HolyLab.github.io/RegisterQD.jl",
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    doctest = :none,  # doctests are run in the test suite via doctest(RegisterQD; manual=false)
)

deploydocs(;
    repo = "github.com/HolyLab/RegisterQD.jl",
    devbranch = "master",
    push_preview = true,
)
