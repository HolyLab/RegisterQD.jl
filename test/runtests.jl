using ImageMagick
using RegisterQD, RegisterMismatch
using Aqua, ExplicitImports, Test

Aqua.test_all(RegisterQD)

# BSplineInterpolation is intentionally accessed as internal API to bypass prefiltering
test_explicit_imports(RegisterQD; ignore=(:BSplineInterpolation,))

include("util.jl")
include("qd_random.jl")
include("qd_standard.jl")
include("gridsearch.jl")
include("initial_tfm.jl")
