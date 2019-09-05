using StaticArrays, Interpolations, LinearAlgebra
using Images, CoordinateTransformations, Rotations
using OffsetArrays
using RegisterMismatch
using RegisterQD

#import BlockRegistration, RegisterOptimize
#using RegisterCore, RegisterPenalty, RegisterDeformation, RegisterMismatch, RegisterFit

using Test, TestImages

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
    tform, mm = qd_affine(fixed, moving, mxshift, SD; maxevals=1000, rtol=0, fvalue=0.0002)
    tfmtest(tfm, tform)

    #with anisotropic sampling
    SD = Matrix(Diagonal([0.5; 1.0]))
    tfm = Translation(@SVector([14.3, 17.8]))∘LinearMap(SD\RotMatrix(0.01)*SD)
    scale = @SMatrix [1.005 0; 0 0.995]
    tfm = AffineMap(tfm.linear*scale, tfm.translation)
    fixed, moving = fixedmov(centered(img), tfm)
    tform, mm = qd_affine(fixed, moving, mxshift, SD; maxevals=1000, rtol=0, fvalue=0.0002)
    tfmtest(tfm, tform)
end #tests with standard images
