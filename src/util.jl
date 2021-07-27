function warp_and_intersect(moving, fixed, tfm::IdentityTransformation)
    if axes(moving) == axes(fixed)
        return moving, fixed
    end
    inds = intersect.(axes(moving), axes(fixed))
    return view(moving, inds...), view(fixed, inds...)
end

function warp_and_intersect(moving, fixed, tfm)
    moving = warp(moving, tfm)
    inds = intersect.(axes(moving), axes(fixed))
    return view(moving, inds...), view(fixed, inds...)
end

"""
    I, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=IdentityTransformation())

Find the best shift `I` (expressed as a tuple) aligning `moving` to `fixed`,
possibly after an initial transformation `initial_tfm` applied to `moving`.
`mm` is the sum-of-square errors at shift `I`.
`mxshift` represents the maximum allowed shift, in pixels.
`thresh` is a threshold on the minimum allowed overlap between `fixed` and the shifted `moving`,
expressed in pixels if `normalization=:pixels` and in sum-of-squared-intensity if `normalization=:intensity`.

One can compute an overall transformation by composing `initial_tfm` with the returned shift `I`.
"""
function best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=IdentityTransformation())
    moving, fixed = warp_and_intersect(moving, fixed, initial_tfm)
    mms = mismatch(fixed, moving, mxshift; normalization=normalization)
    best_i = indmin_mismatch(mms, thresh)
    mm = mms[best_i]
    return best_i.I, ratio(mm, thresh, typemax(eltype(mm)))
end

"""
    itfm = arrayscale(ptfm, SD)

Convert the physical-space transformation `ptfm` into one, `itfm`, that operates on index-space
for arrays.
For example, suppose `ptfm` is a pure 3d rotation, but you want to apply it to an array
in which the sampling along the three axes is (0.5mm, 0.5mm, 2mm). Then by setting
`SD = Diagonal(SVector(1, 1, 4))` (the ratio of scales along each axis),
one obtains an `itfm` suitable for warping the array.

Any translational component of `ptfm` is interpreted in physical-space units, not index-units.

`SD` does not even have to be diagonal, if the array sampling is skewed.
The columns of `SD` should correspond to the physical-coordinate displacement
achieved by shifting by one array element along each axis of the array.
Specifically, `SD[:,j]` should be the physical displacement of one array voxel along dimension `j`.
"""
arrayscale(ptfm::AbstractAffineMap, SD::AbstractMatrix) =
    arrayscale(ptfm, LinearMap(SD))

arrayscale(ptfm::AbstractAffineMap, scale::LinearMap) =
    inv(scale) ∘ ptfm ∘ scale

arrayscale(ptfm::AbstractAffineMap, scale::UniformScaling) = ptfm


"""
    pt = pscale(it, SD)

Convert an index-scaled translation `it` into a physical-space translation `pt`.
See [`arrayscale`](@ref) for more information.
"""
pscale(t::Translation, SD::AbstractMatrix) = pscale(t, LinearMap(SD))
pscale(t::Translation, scale::LinearMap) = Translation(scale(t.translation))
pscale(t::Translation, scale::UniformScaling) = Translation(scale.λ*t.translation)


#returns new minbounds and maxbounds with range sizes change by fac
function scalebounds(minb, maxb, fac::Real)
    orng = maxb.-minb
    newradius = fac.*orng./2
    ctrs = minb.+(orng./2)
    return ctrs.-newradius, ctrs.+newradius
end

"""
minb, maxb = default_linmap_bounds(img::AbstractArray{T,N}; dmax=0.05, ndmax=0.05) where {T,N}

Returns two matrices describing a search space of linear transformation matrices.
(Linear transformation matrices can encode rotations, scaling, shear, etc)
`minb` and  `maxb` contain the minimum and maximum acceptable values of an
NxN transformation matrix.  The center of the space is the identity matrix.
The size of the space can be specified with `dmax` and `ndmax` kwargs.
These represent the maximum (absolute) difference from the identity matrix for elements
along the diagonal and off the diagnonal, respectively.
e.g. `dmax=0.05` implies that diagonal elements can range from 0.95 to 1.05.
The space is centered on the identity matrix
"""
function default_linmap_bounds(img::AbstractArray{T,N}; dmax=0.05, ndmax=0.05) where {T, N}
    deltas = fill(abs(ndmax), N,N)
    for i=1:N
        deltas[i,i] = abs(dmax)
    end
    return Matrix(1.0*I,N,N).-deltas, Matrix(1.0*I,N,N).+deltas
end

"""
m = default_lin_minwidths(img::AbstractArray{T,N}; dmin=1e-3, ndmin=1e-3) where {T,N}

Returns a NxN matrix describing granularity of a search space of linear transformation matrices.
This can be useful for setting the `minwidth` parameter of QuadDIRECT when performing a
full affine registration. `dmin` and `ndmin` set the tolerances for diagonal and
off-diagonal elements of the linear transformation matrix, respectively.
"""
function default_lin_minwidths(img::AbstractArray{T,N}; dmin=1e-5, ndmin=1e-5) where {T, N}
    mat = fill(abs(ndmin), N,N)
    for i=1:N
        mat[i,i] = abs(dmin)
    end
    return mat[:]
end

"""
    θ = default_minrot(ci::CartesianIndices, SD=I; Δc=0.1)

Compute the rotation `θ` that results in largest change in coordinates
of size `Δc` (in pixels) for any index in `ci`.
"""
function default_minrot(ci::CartesianIndices, SD=I; Δc=0.1)
    L = -Inf
    for x in CornerIterator(ci)
        x′ = SD*SVector(Tuple(x))  # position of corner point in physical space
        L = max(L, norm(x′))
    end
    S2 = SD'*SD
    if SD == I
        λ = 1
    else
        F = eigen(S2)
        λ = minimum(F.values)
    end
    ℓ = sqrt(λ)*Δc
    return 2*asin(ℓ/(2*L))
end

default_minwidth_rot(ci::CartesianIndices{2}, SD=I; kwargs...) =
    [default_minrot(ci, SD; kwargs...)]
function default_minwidth_rot(ci::CartesianIndices{3}, SD=I; kwargs...)
    θ = default_minrot(ci, SD; kwargs...)
    return [θ, θ, θ]
end

"""
    getSD(A::AbstractArray)

If your image is not uniformily sampled, use this to get the `SD` matrix, which represents spacing along all axes of an image.

# Examples
```julia-repl
julia> myimage
Normed ImageMeta with:
  data: 3-dimensional AxisArray{N2f14,3,...} with axes:
    :x, 0.0 μm:0.71 μm:6.39 μm
    :l, 0.0 μm:0.71 μm:6.39 μm
    :z, 0.0 μm:6.2 μm:55.8 μm
And data, a 10×10×10 Array{N2f14,3} with eltype Normed{UInt16,14}
  properties:
    imagineheader: <suppressed>

julia> getSD(myimage)
3×3 SArray{Tuple{3,3},Float64,2,9} with indices SOneTo(3)×SOneTo(3):
 0.71  0.0   0.0
 0.0   0.71  0.0
 0.0   0.0   6.2
```
"""
getSD(A::AbstractArray) = getSD(spacedirections(A)) #takes advantage of how spacedirections automatically strips out the time component
function getSD(sd::NTuple{N,Tuple}) where N
    SD = zeros(N,N)
    denom = oneunit(sd[1][1]) #dividing by a unit should remove unit inconsistancies
    for i = 1:N
        for j = 1:N
            SD[i,j] = sd[i][j]/denom
        end
    end
    return SMatrix{N,N}(SD) #returns static matrix for faster type inferrence
end

"""
    imgq = qsmooth(img)
    imgq = qsmooth(T, img)

Create a smoothed version of `img`, smoothing with the kernel of a quadratic B-spline.
Use this on your `fixed` image in preparation for registration, and pass `presmoothed`
as an option. (Do not smooth `moving`.)

`T` allows you to specify the output eltype (default `Float32`).
"""
function qsmooth(::Type{T}, img::AbstractArray{T2,N}) where {T,N,T2}
    kern1 = centered(T[1/8, 3/4, 1/8])   # quadratic B-spline kernel
    return imfilter(img, kernelfactors(ntuple(i->kern1, Val(N))))
end
qsmooth(img::AbstractArray) = qsmooth(Float32, img)

function qinterp(::Type{T}, moving) where T
    widen1(ax) = first(ax)-1:last(ax)+1

    # This sets up an interpolant object *without* using Interpolations.prefilter
    # Prefiltering is a form of "anti-smoothing," meaning that for quadratic interpolation
    # it defines a coefficient array so that, when you interpolate (aka, smooth),
    # it reconstructs the original values. In this case, we *want* interpolation to
    # perform smoothing, because the user has applied the same smoothing to `fixed`
    # by calling `qsmooth`.
    # Build the coefficient array. Quadratic interpolation requires one beyond-the-edge
    # element, which we set to NaN
    nanT = convert(T, NaN)
    mmpad = PaddedView(nanT, of_eltype(T, moving), widen1.(axes(moving)))
    # Create the interpolant, bypassing prefiltering
    return extrapolate(Interpolations.BSplineInterpolation(T, T.(mmpad), BSpline(Quadratic(Free(OnCell()))), axes(moving)), nanT)
end


#     θ = Inf
#     # R = I + [0 -Δθ; Δθ 0] is a rotmtrx for infinitesimal Δθ
#     # `dRdθ` is the "slope" of the rotation, which allows us to compute
#     # the
#     dRdθ = SD \ (@SMatrix([0 -1; 1 0]) * SD)
#     for c in CornerIterator(ci)
#         dcdθs = dRdθ*SVector(Tuple(c))
#         for dcdθ in dcdθs
#             θ = min(θ, abs(Δc/dcdθ))
#         end
#     end
#     return (θ,)
# end

# function default_minwidth_rot(I::CartesianIndices{3}, SD; d=0.1)
#     slice2d(I, i1, i2) = CartesianIndices((I.indices[i1], I.indices[i2]))
#     return (default_minwidth_rot(slice2d(I, 2, 3), SD[2:3, 2:3]; d=d)...,
#             default_minwidth_rot(slice2d(I, 1, 3), SD[[1,3], [1,3]]; d=d)...,
#             default_minwidth_rot(slice2d(I, 1, 2), SD[1:2, 1:2]; d=d)...)
# end

#sets splits based on lower and upper bounds
function _analyze(f, lower, upper; kwargs...)
    splits = ([[lower[i]; lower[i]+(upper[i]-lower[i])/2; upper[i]] for i=1:length(lower)]...,)
    QuadDIRECT.analyze(f, splits, lower, upper; kwargs...)
end

# abs2 was removed in https://github.com/JuliaGraphics/ColorVectorSpace.jl/pull/131
# this is the current recommended replacement
# TODO: follow https://github.com/JuliaGraphics/ColorVectorSpace.jl/issues/157 and adjust for changes
_abs2(c) = mapreducec(v->float(v)^2, +, float(zero(eltype(c))), c)
