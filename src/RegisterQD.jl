module RegisterQD

using Images, CoordinateTransformations, QuadDIRECT
using RegisterMismatch
using RegisterCore #just for indmin_mismatch?
using RegisterDeformation
using Rotations
using Interpolations, CenterIndexedArrays, StaticArrays, OffsetArrays
using LinearAlgebra

using Images.ImageTransformations: CornerIterator

const VecLike = Union{AbstractVector{<:Number}, Tuple{Number, Vararg{Number}}}

include("util.jl")
include("translations.jl")
include("rigid.jl")
include("affine.jl")
include("gridsearch.jl")
include("SD_calculator.jl")

export qd_translate,
        qd_rigid,
        qd_affine,
        arrayscale,
        grid_rotations,
        rotation_gridsearch,
        getSD

# Deprecations
function qd_rigid(fixed, moving, mxshift::VecLike, mxrot::Union{Number,VecLike}, minwidth_rot::VecLike, SD::AbstractMatrix=I; kwargs...)
    error("""
    `qd_rigid` has a new syntax, see the help (`?qd_rigid`) and `NEWS.md`.
    """)
end

function qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD;
                   thresh=0.5*sum(abs2.(fixed[.!(isnan.(fixed))])),
                   initial_tfm=IdentityTransformation(),
                   print_interval=100,
                   kwargs...)
    error("""
    `qd_affine` has a new syntax, see the help (`?qd_affine`) and `NEWS.md`.
    """)
end

end # module
