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

export qd_translate,
        qd_rigid,
        qd_affine,
        arrayscale,
        grid_rotations,
        rotation_gridsearch

# Deprecations
function qd_rigid(fixed, moving, mxshift::VecLike, mxrot::Union{Number,VecLike}, minwidth_rot::VecLike, SD::AbstractMatrix=I; kwargs...)
    error("""
    `qd_rigid` has a new syntax, see the help (`?qd_rigid`).
    In particular, note that the transformation is now returned in *physical*
    units rather than *array-index* units---if you were using something different
    from the identity matrix for `SD`, this is a change in behavior.
    As a consequence, your old results may not be comparable with your new results.
    See also [`arrayscale`](@ref).
    """)
end

function qd_affine(fixed, moving, mxshift, linmins, linmaxs, SD;
                   thresh=0.5*sum(abs2.(fixed[.!(isnan.(fixed))])),
                   initial_tfm=IdentityTransformation(),
                   print_interval=100,
                   kwargs...)
           error("""
           `qd_affine` has a new syntax, see the help (`?qd_affine`).
           """)
end #ad_affine

end # module
