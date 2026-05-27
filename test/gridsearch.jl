using RegisterQD
using RegisterQD.CoordinateTransformations
using RegisterQD.RegisterDeformation
using LinearAlgebra: I
using Test
using OffsetArrays

@testset "Grid search rigid registration" begin
    ## 2D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30, 30)
    b = transform(a, tformtranslate([2.0;0.0]) ∘ tformrotate(pi / 6))
    tfm0 = tformtranslate([-2.0;0.0]) ∘ tformrotate(-pi / 6)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    # SD is now a keyword argument (was positional in v0.x)
    SD2 = Matrix{Float64}(I, 2, 2)
    tfm, mm = RegisterQD.rotation_gridsearch(a, b, (11, 11), [pi / 6], [11]; SD = SD2)
    @test tfm.translation == tfm0.translation
    @test tfm.linear == tfm0.linear

    ## 3D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30, 30, 30)
    b = transform(a, tformtranslate([2.0;0.0;0.0]) ∘ tformrotate([1.0;0;0], pi / 4))
    tfm0 = tformtranslate([-2.0;0.0;0.0]) ∘ tformrotate([1.0;0;0], -pi / 4)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = RegisterQD.rotation_gridsearch(a, b, (3, 3, 3), [pi / 4, pi / 4, pi / 4], [5;5;5])
    @test tfm.translation == tfm0.translation
    @test tfm.linear == tfm0.linear
end

@testset "grid_rotations rounds even rgridsz to odd" begin
    SD = Matrix{Float64}(I, 2, 2)
    rots = @test_logs (:warn, r"rgridsz should be odd") RegisterQD.grid_rotations([pi / 6], [4], SD)
    @test length(rots) == 5  # 4 rounded up to 5
end

@testset "grid_rotations unsupported dimensionality" begin
    SD = Matrix{Float64}(I, 1, 1)
    @test_throws ErrorException RegisterQD.grid_rotations([0.1], [3], SD)
end
