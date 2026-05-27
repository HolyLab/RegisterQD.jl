using ImageMagick
using RegisterQD, RegisterMismatch
using Aqua, Test

Aqua.test_all(RegisterQD)

include("util.jl")
include("qd_random.jl")
include("qd_standard.jl")
include("gridsearch.jl")
include("initial_tfm.jl")
