module RegisterQD

using Images, CoordinateTransformations, QuadDIRECT
using RegisterMismatch
using RegisterCore #just for indmin_mismatch?
using RegisterDeformation
using Rotations
using Interpolations, CenterIndexedArrays, StaticArrays, OffsetArrays
using LinearAlgebra

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
function qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=I; kwargs...)
    error("""
    `qd_rigid` has fundamentally changed. For the new syntax, see the help (`?qd_rigid`).
    In particular, note that the transformation is now returned in *physical*
    units rather than *array-index* units---if you were using something different
    from the identity matrix for `SD`, this is a change in behavior.
    As a consequence, your old results may not be comparable with your new results.
    """)
end

end # module
