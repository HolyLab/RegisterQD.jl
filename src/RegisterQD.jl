module RegisterQD

using Images, CoordinateTransformations, QuadDIRECT
using RegisterMismatch
using RegisterCore #just for indmin_mismatch?
using Rotations
using Interpolations, CenterIndexedArrays, StaticArrays
using AffineTransforms #TODO: remove this

include("util.jl")
include("translations.jl")
include("rigid.jl")
include("affine.jl")
include("gridsearch.jl")

export qd_translate,
        qd_rigid,
        qd_affine,
        grid_rotations,
        rotation_gridsearch

end # module
