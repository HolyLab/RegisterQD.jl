update_SD(SD, tfm::Union{LinearMap, AffineMap}) = update_SD(SD, tfm.linear)
update_SD(SD, tfm::Transformation) = SD
#update_SD(SD::AbstractArray, m::StaticArray) = update_SD(SD, Array(m))
update_SD(SD::AbstractArray, m::AbstractArray) = m\SD*m #TODO found our problem. Why was it written like this?

#rotation only
function rot(theta, img::AbstractArray{T,2}, SD=I) where {T}
    length(theta) == 1 || throw(DimensionMismatch("expected 1 parameters got $(length(thetas))"))
    rotm = SD\RotMatrix(theta...)*SD
    SDS = SMatrix{2,2}(SD)
    return LinearMap(SMatrix{2,2}(rotm))
end
function rot(thetas, img::AbstractArray{T,3}, SD=I) where {T}
    length(thetas) == 3 || throw(DimensionMismatch("expected 3 parameters, got $(length(thetas))"))
    θx, θy, θz = thetas
    rotm = RotMatrix(RotXYZ(θx,θy,θz))
    SDS = SMatrix{3,3}(SD)
    rotm = SDS\rotm*SDS
    return LinearMap(SMatrix{3,3}(rotm))
end #TODO img doesn't seem to be used???

#rotation + shift, fast because it uses fourier method for shift
function rigid_mm_fast(theta, mxshift, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ rot(theta, moving, update_SD(SD, initial_tfm))
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

#rotation + translation
function tfmrigid(params, img::AbstractArray{T,2}, SD=Matrix(1.0*I,2,2)) where {T}
    length(params) == 3 || throw(DimensionMismatch("expected 3 parameters, got $(length(params))"))
    dx, dy, θ = params
    rt = rot(θ, img, SD)
    return Translation(dx, dy) ∘ rt
end
function tfmrigid(params, img::AbstractArray{T,3}, SD=Matrix(1.0*I,3,3)) where {T}
    length(params) == 6 || throw(DimensionMismatch("expected 6 parameters, got $(length(params))"))
    dx, dy, dz, θx, θy, θz =  params
    rt = rot((θx, θy, θz), img, SD)
    #@show SD # TODO SD is warping
    return Translation(dx, dy, dz) ∘ rt
end
#rotation + shift, slow because it warps for every rotation and shift
function rigid_mm_slow(params, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ tfmrigid(params, moving, update_SD(SD, initial_tfm))
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD;
                         initial_tfm=IdentityTransformation(),
                         thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                         kwargs...)
    #note: if a trial rotation results in image overlap < thresh for all possible shifts then QuadDIRECT throws an error
    f(x) = rigid_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper = [mxrot...]
    lower = -upper
    root_coarse, x0coarse = _analyze(f, lower, upper;
                                     minwidth=minwidth_rot, print_interval=100, maxevals=5e4, kwargs..., atol=0, rtol=1e-3)
    box_coarse = minimum(root_coarse)
    tfmcoarse0 = initial_tfm ∘ rot(position(box_coarse, x0coarse), moving, update_SD(SD, initial_tfm)) #why is this second one a valid rotation?
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=tfmcoarse0)
    tfmcoarse = tfmcoarse0 ∘ Translation(best_shft)
    return tfmcoarse, mm
end

function qd_rigid_fine(fixed, moving, mxrot, minwidth_rot, SD;
                       initial_tfm=IdentityTransformation(),
                       thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                       kwargs...)
    f(x) = rigid_mm_slow(x, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper_shft = fill(2.0, ndims(fixed))
    upper_rot = mxrot
    upper = vcat(upper_shft, upper_rot)
    lower = -upper
    minwidth_shfts = fill(0.005, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_rot)
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    # @show isrotation(SD*initial_tfm.linear*inv(SD))
    # @show isrotation(SD*tfmrigid(position(box, x0), moving, update_SD(SD, initial_tfm)).linear*inv(SD))
    tfmfine = initial_tfm ∘ tfmrigid(position(box, x0), moving, update_SD(SD, initial_tfm))
    #TODO change this to just SD instead of update SD? the analyze function may still be off.
    # tfmfine = initial_tfm ∘ tfmrigid(position(box, x0), moving, SD)
    # @show isrotation(SD*tfmfine.linear*inv(SD))
    return tfmfine, value(box), root, x0, box, update_SD(SD, initial_tfm), tfmrigid(position(box, x0), moving, update_SD(SD, initial_tfm))
end

"""
`tform, mm = qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=I;  thresh=thresh, initial_tfm=IdentityTransformation(), kwargs...)`
optimizes a rigid transformation (rotation + shift) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy. `kwargs...` can include any
keyword argument that can be passed to `QuadDIRECT.analyze`. It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).  If you provide `rtol` and/or `atol` they will apply only to the second (fine) step of the registration;
the user may not adjust these criteria for the coarse step.

The rotation returned will be centered on the origin-of-coordinates, i.e. (0,0) for a 2D image.  Usually it is more natural to consider rotations
around the center of the image.  If you would like `mxrot` and the returned rotation to act relative to the center of the image, then you must
move the origin to the center of the image by calling `centered(img)` from the `ImageTransformations` package.  Call `centered` on both the
fixed and moving image to generate the `fixed` and `moving` that you provide as arguments.  If you later want to apply the returned transform
to an image you must remember to call `centered` on that image as well.  Alternatively you can re-encode the transformation in terms of a
different origin by calling `recenter(tform, newctr)` where `newctr` is the displacement of the new center from the old center.

Use `SD` if your axes are not uniformly sampled, for example `SD = diagm(voxelspacing)` where `voxelspacing`
is a vector encoding the spacing along all axes of the image. `thresh` enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.
"""
function qd_rigid(fixed, moving, mxshift, mxrot, minwidth_rot, SD=Matrix(1.0*I, ndims(fixed), ndims(fixed));
                  thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                  initial_tfm=IdentityTransformation(),
                  print_interval=100,
                  kwargs...)
    fixed, moving = float(fixed), float(moving)
    mxrot = [mxrot...]
    print_interval < typemax(Int) && print("Running coarse step\n")
    tfm_coarse, mm_coarse = qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot, SD; initial_tfm=initial_tfm, thresh=thresh, print_interval=print_interval, kwargs...)
    print_interval < typemax(Int) && print("Running fine step\n")
    final_tfm, mm_fine = qd_rigid_fine(fixed, moving, mxrot./2, minwidth_rot, SD; initial_tfm=tfm_coarse, thresh=thresh, print_interval=print_interval, kwargs...)
    return final_tfm, mm_fine
end
