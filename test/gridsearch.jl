using AffineTransforms #TODO: remove
using RegisterQD
using Test

@testset "Grid search rigid registration" begin
    ## 2D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30)
    b = AffineTransforms.transform(a, tformtranslate([2.0;0.0]) * tformrotate(pi/6))
    tfm0 = tformtranslate([-2.0;0.0]) * tformrotate(-pi/6)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = rotation_gridsearch(a, b, [11;11], [pi/6], [11])
    @test tfm.offset == tfm0.offset
    @test tfm.scalefwd == tfm0.scalefwd

    ## 3D
    #note: if a is much smaller than this then it won't find the correct answer due to the mismatch normalization
    a = rand(30,30,30)
    b = AffineTransforms.transform(a, tformtranslate([2.0;0.0;0.0]) * tformrotate([1.0;0;0], pi/4))
    tfm0 = tformtranslate([-2.0;0.0;0.0]) * tformrotate([1.0;0;0], -pi/4)
    #note: maxshift must be GREATER than the true shift in order to find the true shift
    tfm, mm = rotation_gridsearch(a, b, [3;3;3], [pi/4, pi/4, pi/4], [5;5;5])
    @test tfm.offset == tfm0.offset
    @test tfm.scalefwd == tfm0.scalefwd
end
