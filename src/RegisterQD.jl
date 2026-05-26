module RegisterQD

using ImageCore, ImageTransformations, ImageFiltering
using CoordinateTransformations
using QuadDIRECT
using RegisterMismatchCommon
using RegisterCore
using RegisterDeformation, PaddedViews, MappedArrays
using Rotations
using Interpolations, CenterIndexedArrays, StaticArrays, OffsetArrays
using LinearAlgebra

using ImageTransformations: CornerIterator

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
    rotation_gridsearch,
    getSD,
    qsmooth

end # module
