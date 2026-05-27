module RegisterQD

using CenterIndexedArrays: CenterIndexedArrays
using CoordinateTransformations: CoordinateTransformations, AbstractAffineMap, AffineMap,
    IdentityTransformation, LinearMap, Translation
using ImageCore: ImageCore, spacedirections
using ImageFiltering: ImageFiltering, centered, imfilter, kernelfactors
using ImageTransformations: ImageTransformations, warp
using Interpolations: Interpolations, BSpline, Free, OnCell, Quadratic, extrapolate
using LinearAlgebra: LinearAlgebra, I, UniformScaling, eigen, norm
using MappedArrays: MappedArrays, of_eltype
using OffsetArrays: OffsetArrays, OffsetArray
using PaddedViews: PaddedViews, PaddedView
using QuadDIRECT: QuadDIRECT, value
using RegisterCore: RegisterCore, argmin_mismatch, ratio
using RegisterDeformation: RegisterDeformation, tformeye, tformrotate, tformtranslate, transform
using RegisterMismatchCommon: RegisterMismatchCommon, mismatch, mismatch_zeroshift
using Rotations: Rotations, RotMatrix, isrotation
using StaticArrays: StaticArrays, @SMatrix, SMatrix, SVector

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
