using RegisterQD
using LinearAlgebra
using ImageMagick
using Distributions
using RegisterQD.StaticArrays
using RegisterQD.Interpolations
using RegisterQD.Images
using RegisterQD.CoordinateTransformations
using RegisterQD.Rotations
using RegisterQD.OffsetArrays
using RegisterQD.RegisterMismatch

using Test, TestImages
using Random

#Helper to generate test image pairs
function fixedmov(img, tfm)
    img = float(img)
    img2 = warp(img,tfm)
    inds = OffsetArrays.IdentityUnitRange.(intersect.(axes(img), axes(img2)))
    fixed = img[inds...]
    moving = img2[inds...]
    return fixed, moving
end

#helpers to convert Transformations to AffineMaps
to_affine(tfm::Translation) = AffineMap(Matrix{Float64}(LinearAlgebra.I, length(tfm.translation), length(tfm.translation)), tfm.translation)
to_affine(tfm::LinearMap) = AffineMap(Matrix{Float64}(LinearAlgebra.I, length(tfm.translation), length(tfm.translation)), tfm.translation)
to_affine(tfm::AffineMap) = tfm

#Helper to test that a found transform is (roughly) the inverse of the original transform
function tfmtest(tfm, tfminv)
    comp = to_affine(tfm ∘ tfminv)  #should be the identity transform
    diagtol = 0.005
    offdiagtol = 0.005
    vtol = 0.1
    @test all(x->(1-diagtol < x < 1+diagtol), diag(comp.linear))
    @test all(x->(-offdiagtol < x < offdiagtol), comp.linear.-Matrix(Diagonal(diag(comp.linear))))
    @test all(abs.(comp.translation) .< vtol)
end

# tests with standard images
# (Also, unlike the tests above these tests set up the problem so that the correct
# answer to is the inverse of an input transformation.  This seems to catch
# a different set of errors than the tests above)
@testset "QuadDIRECT tests with standard images" begin
    img = testimage("cameraman");

    #Translation (subpixel)
    tfm = Translation(@SVector([14.3, 17.6]))
    fixed, moving = fixedmov(img, tfm)
    mxshift = (100,100) #make sure this isn't too small
    tform, mm = qd_translate(fixed, moving, mxshift; maxevals=1000, rtol=0, fvalue=0.0003)
    tfmtest(tfm, tform)

    #Rigid transform
    SD = Matrix{Float64}(LinearAlgebra.I, 2, 2)
    tfm = Translation(@SVector([14, 17]))∘LinearMap(RotMatrix(0.3)) #no distortion for now
    fixed, moving = fixedmov(centered(img), tfm)
    mxshift = (100,100) #make sure this isn't too small
    mxrot = (0.5,)
    minwidth_rot = fill(0.002, 3)
    tform, mm = qd_rigid(fixed, moving, mxshift, mxrot; SD=SD, maxevals=1000, rtol=0, fvalue=0.0002)
    tfmtest(tfm, tform)
    #with anisotropic sampling
    SD = Matrix(Diagonal([0.5; 1.0]))
    tfm = Translation(@SVector([14.3, 17.8]))∘LinearMap(SD\RotMatrix(0.3)*SD)
    fixed, moving = fixedmov(centered(img), tfm)
    tform, mm = qd_rigid(fixed, moving, mxshift, mxrot; SD=SD, maxevals=1000, rtol=0, fvalue=0.0002)
    tfmtest(tfm, arrayscale(tform, SD))

    #Affine transform
    tfm = Translation(@SVector([14, 17]))∘LinearMap(RotMatrix(0.01))
    #make it harder with nonuniform scaling
    scale = @SMatrix [1.005 0; 0 0.995]
    SD = Matrix{Float64}(LinearAlgebra.I, 2, 2)
    tfm = AffineMap(tfm.linear*scale, tfm.translation)
    mxshift = (100,100) #make sure this isn't too small
    fixed, moving = fixedmov(centered(img), tfm)
    tform, mm = qd_affine(fixed, moving, mxshift; SD = SD, maxevals=1000, rtol=0, fvalue=0.0002)
    tfmtest(tfm, tform)

    #with anisotropic sampling
    SD = Matrix(Diagonal([0.5; 1.0]))
    tfm = Translation(@SVector([14.3, 17.8]))∘LinearMap(RotMatrix(0.1)) #Translation(@SVector([14.3, 17.8]))∘LinearMap(SD\RotMatrix(0.01)*SD)
    scale = @SMatrix [1.005 0; 0 0.995]
    tfm = AffineMap(tfm.linear*scale, tfm.translation)
    tfm = arrayscale(tfm, SD)
    fixed, moving = fixedmov(centered(img), tfm)
    tform, mm = qd_affine(fixed, moving, mxshift; SD = SD, maxevals=10000, rtol=0, fvalue=0.0002, ndmax = 0.25)
    tform2 = arrayscale(tform, SD)
    tfmtest(tfm, tform2)
end #tests with standard images

@testset "Quadratic interpolation (issue #7)" begin
    samplefrom(n) = rand(Poisson(n))

    Random.seed!(222)
    img = restrict(restrict(testimage("cameraman")))[2:end-1,2:end-1]
    # Convert to "photons" so we can mimic shot noise
    np = 100  # maximum number of photons per pixel
    img = round.(Int, np.*gray.(img))
    fixed  = samplefrom.(img)
    moving = samplefrom.(img)
    ff = qsmooth(fixed)

    tform, mm = qd_translate(fixed, moving, (5, 5); print_interval=typemax(Int))
    tformq, mmq = qd_translate(ff, moving, (5, 5); presmoothed=true, print_interval=typemax(Int))
    @test all(abs.(tformq.translation) .< abs.(tform.translation))

    tform, mm = qd_rigid(fixed, moving, (5, 5), 0.1; print_interval=typemax(Int))
    tformq, mmq = qd_rigid(ff, moving, (5, 5), 0.1; presmoothed=true, print_interval=typemax(Int))
    @test norm(tformq.linear-I) < norm(tform.linear-I)
    @test norm(tformq.translation) < norm(tform.translation)

    tform, mm = qd_affine(fixed, moving, (5, 5); print_interval=typemax(Int))
    tformq, mmq = qd_affine(ff, moving, (5, 5); presmoothed=true, print_interval=typemax(Int))
    @test mmq < mm

    # Test that we exactly reconstruct `qsmooth` with `presmoothed=true`
    tformq, mmq = qd_translate(ff, fixed, (5, 5); presmoothed=true, print_interval=typemax(Int))
    @test all(iszero, tformq.translation)
    @test mmq < 1e-10
    tformq, mmq = qd_rigid(ff, fixed, (5, 5), 0.1; presmoothed=true, print_interval=typemax(Int))
    @test mmq < 1e-8
    tformq, mmq = qd_affine(ff, fixed, (5, 5); presmoothed=true, print_interval=typemax(Int))
    @test mmq < 1e-6   # on 32-bit systems this can't be 1e-8, not quite sure why
end
