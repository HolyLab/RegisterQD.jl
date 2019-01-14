function warp_and_intersect(moving, fixed, tfm)
    if tfm == IdentityTransformation()
        if axes(moving) == axes(fixed)
            return moving, fixed
        end
    else
        moving = warp(moving, tfm)
    end
    inds = intersect.(axes(moving), axes(fixed))
    #TODO: use views after BlockRegistration #83 on Github is addressed
    return moving[inds...], fixed[inds...]
end


#Finds the best shift aligning moving to fixed, possibly after an initial transformation `initial_tfm`
#The shift returned should be composed with initial_tfm later to create the full transform
function best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=IdentityTransformation())
    moving, fixed = warp_and_intersect(moving, fixed, initial_tfm)
    mms = mismatch(fixed, moving, mxshift; normalization=normalization)
    best_i = indmin_mismatch(mms, thresh)
    return best_i.I, ratio(mms[best_i], 0.0, Inf)
end

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

#sets splits based on lower and upper bounds
function _analyze(f, lower, upper; kwargs...)
    splits = ([[lower[i]; lower[i]+(upper[i]-lower[i])/2; upper[i]] for i=1:length(lower)]...,)
    QuadDIRECT.analyze(f, splits, lower, upper; kwargs...)
end
