using TestImages
using AxisArrays
using LinearAlgebra


@testset "getSD" begin
    A2 = rand(10,10)
    @test getSD(A2) == diagm(ones(2))

    A3 = rand(10,10,10)
    @test getSD(A3) == diagm(ones(3))

    A5 = rand(10,10,10,10,10)
    @test getSD(A5) == diagm(ones(5))

    Ax3 = AxisArray(A3, 1:1:10, 1:2:20, 1:3:30)
    @test getSD(Ax3) == diagm([1.0,2.0,3.0])

    mri = testimage("mri-stack.tif")
    @test getSD(mri) == diagm([1.0, 1.0, 5.0])
end

#TODO it would be nice to add tests for skewed SD,
