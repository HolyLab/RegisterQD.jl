using RegisterQD
using RegisterQD.CoordinateTransformations
using RegisterQD.RegisterDeformation
using Test

@testset "Grid search rigid registration" begin
    ## 2D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30)
    b = transform(a, tformtranslate([2.0;0.0]) ∘ tformrotate(pi/6))
    tfm0 = tformtranslate([-2.0;0.0]) ∘ tformrotate(-pi/6)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = RegisterQD.rotation_gridsearch(a, b, (11,11), [pi/6], [11])
    @test tfm.translation == tfm0.translation
    @test tfm.linear == tfm0.linear

    ## 3D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30,30)
    b = transform(a, tformtranslate([2.0;0.0;0.0]) ∘ tformrotate([1.0;0;0], pi/4))
    tfm0 = tformtranslate([-2.0;0.0;0.0]) ∘ tformrotate([1.0;0;0], -pi/4)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = RegisterQD.rotation_gridsearch(a, b, (3,3,3), [pi/4, pi/4, pi/4], [5;5;5])
    @test tfm.translation == tfm0.translation
    @test tfm.linear == tfm0.linear
end
