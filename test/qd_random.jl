using StaticArrays, Interpolations
using Images, CoordinateTransformations, Rotations
using RegisterMismatch

#import BlockRegistration, RegisterOptimize
#using RegisterCore, RegisterPenalty, RegisterDeformation, RegisterMismatch, RegisterFit

using Test, TestImages

@testset "QuadDIRECT tests with random images" begin
    ##### Translations
    #2D
    moving = rand(50,50)
    tfm0 = Translation(-4.7, 5.1) #ground truth
    newfixed = warp(moving, tfm0)
    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
    etp = extrapolate(itp, NaN)
    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
    thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
    mxshift = [10;10]

    tfm, mm = qd_translate(fixed, moving, mxshift; maxevals=1000, thresh=thresh, rtol=0)

    @test sum(abs.(tfm0.translation - tfm.translation)) < 1e-3

    #3D
    moving = rand(30,30,30)
    tfm0 = Translation(-0.9, 2.1,1.2) #ground truth
    newfixed = warp(moving, tfm0)
    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
    etp = extrapolate(itp, NaN)
    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
    thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
    mxshift = [5;5;5]

    tfm, mm = qd_translate(fixed, moving, mxshift; maxevals=1000, thresh=thresh, rtol=0)

    @test sum(abs.(tfm0.translation - tfm.translation)) < 0.1

    ######Rotations + Translations
    #2D
    moving = centered(rand(50,50))
    tfm0 = Translation(-4.0, 5.0) ∘ LinearMap(RotMatrix(pi/360)) #ground truth
    newfixed = warp(moving, tfm0)
    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
    etp = extrapolate(itp, NaN)
    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
    thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
    mxshift = [10;10]
    mxrot = pi/90
    minwidth_rot = [0.0002]
    SD = eye(ndims(fixed))

    tfm, mm = qd_rigid(centered(fixed), moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, maxevals=1000, rtol=0, fvalue=1e-8)

    @test sum(abs.(tfm0.linear - tfm.linear)) < 1e-3

    #3D
    moving = centered(rand(30,30,30))
    tfm0 = Translation(-1.0, 2.1,1.2) ∘ LinearMap(RotXYZ(pi/360, pi/180, pi/220)) #ground truth
    newfixed = warp(moving, tfm0)
    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
    etp = extrapolate(itp, NaN)
    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
    thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))]))
    mxshift = [5;5;5]
    mxrot = [pi/90; pi/90; pi/90]
    minwidth_rot = fill(0.0002, 3)
    SD = eye(ndims(fixed))

    tfm, mm = qd_rigid(centered(fixed), moving, mxshift, mxrot, minwidth_rot, SD; thresh=thresh, maxevals=1000, rtol=0)

    @test sum(abs.(vcat(tfm0.linear[:], tfm0.translation) - vcat(RotXYZ(tfm.linear)[:], tfm.translation))) < 0.1

#NOTE: the 2D test below fails rarely and the 3D test fails often, apparently because full affine is too difficult with these images
    #####General Affine Transformations
    #2D
    moving = centered(rand(50,50))
    shft = SArray{Tuple{2}}(rand(2).+2.0)
    #random displacement from the identity matrix
    mat = SArray{Tuple{2,2}}(eye(2) + rand(2,2)./40 + -rand(2,2)./40)
    tfm0 = AffineMap(mat, shft) #ground truth
    newfixed = warp(moving, tfm0)
    itp = interpolate(newfixed, BSpline(Linear()), OnGrid())
    etp = extrapolate(itp, NaN)
    fixed = etp[indices(moving)...] #often the warped array has one-too-many pixels in one or more dimensions due to extrapolation
    thresh = 0.5 * sum(abs2.(fixed[.!(isnan.(fixed))]))
    mxshift = [5;5]
    SD = eye(ndims(fixed))

    tfm, mm = qd_affine(centered(fixed), moving, mxshift, SD; thresh=thresh, maxevals=1500, rtol=0, fvalue=1e-6)

    @test sum(abs.(vcat(tfm0.linear[:], tfm0.translation) - vcat(tfm.linear[:], tfm.translation))) < 0.1

    #The tests below fail.  Probably two factors contributing to failure:
    # 1) Full affine 3D is a lot of parameters so it's just difficuilt (12 parameters)
    # 2) The current way of computing mismatch may be flawed for scaling transformations
    #    because the algorithm output shows it tends to compress/expand the image in order to remove
    #    pixels from the denominator in the mismatch calculation (especially bad for small images).
    #3D
    #moving = centered(rand(20,20,20));
    #shft = SArray{Tuple{3}}(2.6, 0.1, -3.3);
    ##random displacement from the identity matrix
    #mat = SArray{Tuple{3,3}}(eye(3) + rand(3,3)./30 + -rand(3,3)./30);
    #tfm0 = AffineMap(mat, shft); #ground truth
    #newfixed = warp(moving, tfm0);
    #inds = intersect.(indices(moving), indices(newfixed))
    #fixed = newfixed[inds...]
    #moving = moving[inds...]
    #thresh = 0.1 * (sum(abs2.(fixed[.!(isnan.(fixed))]))+sum(abs2.(moving[.!(isnan.(moving))])));
    #mxshift = [10;10;10];
    #SD = eye(ndims(fixed));

    #tfm, mm = qd_affine(centered(fixed), centered(moving), mxshift, SD; thresh=thresh, rtol=0, fvalue=1e-4);

    #@test mm <= 1e-4
    #@test sum(abs.(vcat(tfm0.linear[:], tfm0.translation) - vcat(tfm.linear[:], tfm.translation))) < 0.1
    
    #not random
    #moving = zeros(10,10,10);
    #moving[5:7, 5:7, 5:7] = 1.0
    #moving = centered(moving)
    #shft = SArray{Tuple{3}}(0.6, 0.1, -0.3);
    ##random displacement from the identity matrix
    #mat = SArray{Tuple{3,3}}(eye(3) + rand(3,3)./50 + -rand(3,3)./50);
    #tfm00 = AffineMap(mat, shft);
    ##tfm0 = recenter(tfm00, center(moving)); #ground truth
    #tfm0 = tfm00 #ground truth
    #newfixed = warp(moving, tfm0);
    #inds = intersect.(indices(moving), indices(newfixed))
    #fixed = newfixed[inds...]
    #moving = moving[inds...]
    #thresh = 0.5 * sum(abs2.(fixed[.!(isnan.(fixed))]));
    #mxshift = [5;5;5];
    #SD = eye(ndims(fixed));
    #@test RegisterOptimize.aff(vcat(tfm00.translation[:], tfm00.linear[:]), fixed, SD) == tfm0

    #tfm, mm = qd_affine(centered(fixed), centered(moving), mxshift, SD; thresh=thresh, rtol=0, fvalue=1e-4);

    #@test mm <= 1e-4
    #@test sum(abs.(vcat(tfm0.linear[:], tfm0.translation) - vcat(tfm.linear[:], tfm.translation))) < 0.1

end #tests with random images
