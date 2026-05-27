using ImageMagick
using RegisterQD, RegisterMismatch
using AxisArrays
using Aqua, Documenter, ExplicitImports, Test

Aqua.test_all(RegisterQD)

# BSplineInterpolation is intentionally accessed as internal API to bypass prefiltering
test_explicit_imports(RegisterQD; ignore=(:BSplineInterpolation,))

include("util.jl")
include("qd_random.jl")
include("qd_standard.jl")
include("gridsearch.jl")
include("initial_tfm.jl")

DocMeta.setdocmeta!(
    RegisterQD, :DocTestSetup,
    :(using RegisterQD, RegisterMismatch, AxisArrays, CoordinateTransformations, LinearAlgebra);
    recursive = true,
)
@testset "Doctests" begin
    doctest(RegisterQD; manual=false)
end
