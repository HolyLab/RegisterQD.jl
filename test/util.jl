using ImageMagick
using TestImages
using RegisterQD
using RegisterQD.CoordinateTransformations: IdentityTransformation
using LinearAlgebra
using ImageMetadata
import AxisArrays
using AxisArrays: AxisArray, Axis
using Unitful: μm, mm, cm, km, s
using OffsetArrays

@testset "default_minwidth_rot" begin
    img = rand(3, 10)
    ci = CartesianIndices(img)
    θ = RegisterQD.default_minrot(ci)
    @test θ ≈ 0.01 rtol = 0.1
    θ = RegisterQD.default_minrot(ci, [1 0; 0 2])
    @test θ ≈ 0.005 rtol = 0.1
    θ = RegisterQD.default_minrot(ci, [3 0; 0 1])
    @test θ ≈ 0.007 rtol = 0.1
    θ = RegisterQD.default_minrot(ci, [10 0; 0 1])
    @test θ ≈ 0.01 / 3 rtol = 0.1
    img = rand(3, 10, 5)
    ci = CartesianIndices(img)
    θ = RegisterQD.default_minrot(ci)
    @test θ ≈ 0.1 / sqrt(3^2 + 10^2 + 5^2) rtol = 1.0e-3
    # array overload matches the CartesianIndices form
    SD = [1 0 0; 0 2 0; 0 0 3]
    @test RegisterQD.default_minrot(img) == RegisterQD.default_minrot(CartesianIndices(img))
    @test RegisterQD.default_minrot(img, SD) == RegisterQD.default_minrot(CartesianIndices(img), SD)
    @test RegisterQD.default_minrot(img, SD; Δc = 0.2) == RegisterQD.default_minrot(CartesianIndices(img), SD; Δc = 0.2)
end

@testset "getSD" begin
    #test that getSD deals with arbitrary dimensions
    A2 = rand(10, 10)
    @test getSD(A2) == I

    A3 = rand(10, 10, 10)
    @test getSD(A3) == I

    A5 = rand(10, 10, 10, 10, 10)
    @test getSD(A5) == I

    Ax3 = AxisArray(A3, 1:1:10, 1:2:20, 1:3:30)
    @test getSD(Ax3) == Diagonal([1.0, 2.0, 3.0])

    #test that getSD works with test images
    mri = testimage("mri-stack.tif")
    @test getSD(mri) == Diagonal([1.0, 1.0, 5.0])

    #test that getSD deals with images with arbitrary space directions
    skewed = ImageMeta(rand(10, 10, 10))
    skewmatrix = rand(3, 3)
    skewed.spacedirections = (Tuple(skewmatrix[1, :]), Tuple(skewmatrix[2, :]), Tuple(skewmatrix[3, :]))
    getSD(skewed) == skewmatrix

    #test that getSD can reconcile units of different magnitudes
    badsampling = AxisArray(rand(10, 10, 10), (:x, :y, :z), (1mm, 2km, 3.4cm))
    badsampling = ImageMeta(badsampling)
    @test getSD(badsampling) == Diagonal([1, 2.0e6, 34])

    #test that getSD ignores the time-axis
    timedarray = AxisArray(rand(10, 10, 10), (:x, :y, :time), (1μm, 1μm, 1s))
    @test size(getSD(timedarray)) == (2, 2)
end

@testset "qsmooth eltype" begin
    img64 = rand(Float64, 8, 8)
    @test eltype(qsmooth(img64)) === Float64
    img32 = rand(Float32, 8, 8)
    @test eltype(qsmooth(img32)) === Float32
    # explicit T still widens the compute/output eltype
    @test eltype(qsmooth(Float64, img32)) === Float64
end

@testset "warp_and_intersect IdentityTransformation with mismatched axes" begin
    moving = OffsetArray(rand(5, 5), 1:5, 1:5)
    fixed  = OffsetArray(rand(5, 5), 3:7, 3:7)
    vm, vf = RegisterQD.warp_and_intersect(moving, fixed, IdentityTransformation())
    @test Base.axes(vm) == Base.axes(vf)
    @test length.(Base.axes(vm)) == (3, 3)  # intersection of 1:5 and 3:7 has 3 elements per dim
end
