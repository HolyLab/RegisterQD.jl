"""
    RegisterQD

Image registration using the [QuadDIRECT](https://github.com/timholy/QuadDIRECT.jl)
global optimization algorithm.

The three main entry points are:
- [`qd_translate`](@ref): optimize a pure translation
- [`qd_rigid`](@ref): optimize a rigid transformation (rotation + translation)
- [`qd_affine`](@ref): optimize a full affine transformation

All three return `(tform, mm)` where `tform` is a
[CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl)
transform object and `mm` is the residual mismatch value (lower is better).

!!! note
    A mismatch backend such as
    [RegisterMismatch.jl](https://github.com/HolyLab/RegisterMismatch.jl) must be
    loaded (`using RegisterMismatch`) before calling any registration function.

# Utilities
- [`arrayscale`](@ref): convert a physical-space transform to array-index space
- [`getSD`](@ref): extract the spatial-directions matrix from an annotated image
- [`qsmooth`](@ref): pre-smooth an image for registration
- [`grid_rotations`](@ref) / [`rotation_gridsearch`](@ref): coarse rotation grid search
"""
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
