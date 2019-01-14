"""
`rotations = grid_rotations(maxradians, rgridsz, SD)` generates
a set of rotations (AffineMap) useful for a gridsearch of
possible rotations to align a pair of images.

`maxradians` is either a single maximum angle (in 2D) or a set of
Euler angles (in 3D and higher). `rgridsz` is one or more integers
specifying the number of gridpoints to search in each of the rotation
axes, corresponding with entries in `maxradians`. `SD` is a matrix
specifying the sample spacing.

e.g. `grid_rotations((pi/8,pi/8,pi/8), (3,3,3), Matrix{Float64}(I,3,3))` would
return an array of 27 rotations with 3 possible angles for each
Euler axis: -pi/8, 0, pi/8. Passing `Matrix{Float64}(I,3,3)` for SD indicates
that the resulting  transforms are meant to be applied to an image with isotropic
pixel spacing.
"""
function grid_rotations(maxradians, rgridsz, SD)
    rgridsz = [rgridsz...]
    maxradians = [maxradians...]
    nd = size(SD,1)
    if !all(isodd, rgridsz)
        @warn("rgridsz should be odd; rounding up to the next odd integer")
    end
    for i = 1:length(rgridsz)
        if !isodd(rgridsz[i])
            rgridsz[i] = max(round(Int, rgridsz[i]) + 1, 1)
        end
    end
    grid_radius = map(x->div(x,2), rgridsz)
    if nd > 2
        gridcoords = [range(-grid_radius[x]*maxradians[x], stop=grid_radius[x]*maxradians[x], length=rgridsz[x]) for x=1:nd]
        rotation_angles = Iterators.product(gridcoords...)
    else
        rotation_angles = range(-grid_radius[1]*maxradians[1], stop=grid_radius[1]*maxradians[1], length=rgridsz[1])
    end
    axmat = Matrix{Float64}(I,nd,nd)
    axs = map(x->axmat[:,x], 1:nd)
    tfeye = tformeye(nd)
    output = typeof(tfeye)[]
    for ra in rotation_angles
        if nd > 2
            euler_rots = map(x->tformrotate(x...), zip(axs, ra))
            rot = foldr(*, tfeye, euler_rots)
        elseif nd == 2
            rot = tformrotate(ra)
        else
            error("Unsupported dimensionality")
        end
        push!(output, AffineMap(SD*rot.scalefwd/SD , zeros(nd))) #account for sample spacing
    end
    return output
end

"""
`best_tform, best_mm = rotation_gridsearch(fixed, moving, maxshift, maxradians, rgridsz, SD =Matrix{Float64}(I,ndims(fixed),ndims(fixed))))`
Tries a grid of rotations to align `moving` to `fixed`.  Also calculates the best translation (`maxshift` pixels
or less) to align the images after performing the rotation. Returns an AffineMap that captures both the
best rotation and shift out of the values searched, along with the mismatch value after applying that transform (`best_mm`).

For more on how the arguments `maxradians`, `rgridsz`, and `SD` influence the search, see the documentation for
`grid_rotations`.
"""
function rotation_gridsearch(fixed, moving, maxshift, maxradians, rgridsz, SD = Matrix{Float64}(I,ndims(fixed),ndims(fixed)))
    rgridsz = [rgridsz...]
    nd = ndims(moving)
    @assert nd == ndims(fixed)
    rots = grid_rotations(maxradians, rgridsz, SD)
    best_mm = Inf
    best_rot = tformeye(ndims(moving))
    best_shift = zeros(nd)
    for rot in rots
        new_moving = transform(moving, rot)
        #calc mismatch
        #mm = mismatch(fixed, new_moving, maxshift; normalization=:pixels)
        mm = mismatch(fixed, new_moving, maxshift)
        thresh = 0.1*maximum(x->x.denom, mm)
        best_i = indmin_mismatch(mm, thresh)
        cur_best =ratio(mm[best_i], 0.0)
        if cur_best < best_mm
            best_mm = cur_best
            best_rot = rot
            best_shift = [best_i.I...]
        end
    end
    return tformtranslate(best_shift) * best_rot, best_mm
end
