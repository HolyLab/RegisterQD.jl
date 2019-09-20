using ImageMagick
using TestImages
using LinearAlgebra
using AxisArrays, ImageMetadata
using Unitful: μm, mm, cm, km, s

@testset "default_minwidth_rot" begin
    img = rand(3, 10)
    ci = CartesianIndices(img)
    θ = RegisterQD.default_minrot(ci)
    @test θ ≈ 0.01 rtol=0.1
    θ = RegisterQD.default_minrot(ci, [1 0; 0 2])
    @test θ ≈ 0.005 rtol=0.1
    θ = RegisterQD.default_minrot(ci, [3 0; 0 1])
    @test θ ≈ 0.007 rtol=0.1
    θ = RegisterQD.default_minrot(ci, [10 0; 0 1])
    @test θ ≈ 0.01/3 rtol=0.1
    img = rand(3, 10, 5)
    ci = CartesianIndices(img)
    θ = RegisterQD.default_minrot(ci)
    @test θ ≈ 0.1/sqrt(3^2 + 10^2 + 5^2) rtol=1e-3
end

@testset "getSD" begin
    #test that getSD deals with arbitrary dimensions
    A2 = rand(10,10)
    @test getSD(A2) == diagm(ones(2))

    A3 = rand(10,10,10)
    @test getSD(A3) == diagm(ones(3))

    A5 = rand(10,10,10,10,10)
    @test getSD(A5) == diagm(ones(5))

    Ax3 = AxisArray(A3, 1:1:10, 1:2:20, 1:3:30)
    @test getSD(Ax3) == diagm([1.0,2.0,3.0])

    #test that getSD works with test images
    mri = testimage("mri-stack.tif")
    @test getSD(mri) == diagm([1.0, 1.0, 5.0])

    #test that getSD deals with images with arbitrary space directions
    skewed = ImageMeta(rand(10,10,10))
    skewmatrix = rand(3,3)
    skewed["spacedirections"] = (Tuple(skewmatrix[1,:]), Tuple(skewmatrix[2,:]), Tuple(skewmatrix[3,:]))
    getSD(skewed) == skewmatrix

    #test that getSD can reconcile units of different magnitudes
    badsampling = AxisArray(rand(10,10,10), (:x,:y,:z), (1mm, 2km, 3.4cm))
    badsampling = ImageMeta(badsampling)
    @test getSD(badsampling) == diagm([1, 2e6, 34])

    #test that getSD ignores the time-axis
    timedarray = AxisArray(rand(10,10,10), (:x, :y, :time), (1μm, 1μm, 1s))
    @test size(getSD(timedarray)) == (2,2)
end



#TODO add a testset for other support functions
#rotations
#arrayscale
#pscale
#restrict
