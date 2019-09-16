using StaticArrays, Interpolations, LinearAlgebra
using Images, CoordinateTransformations, Rotations
using RegisterQD

g = 0.2:0.2:1.2
gradcube = g .* reshape(g, 1, 6) .* reshape(g, 1, 1, 6)
gradcube[1,4,2] = 0
gradcube[4,5,2] = 0 #break up any potential rotational symmetry

mxshift = (1,1,1);
mxrot = (0.01,0.01,0.01);
minwidth_rot = fill(0.002,3);

EYE =Matrix(1.0*I, 3,3)

@testset "initial_tfm improves rigid rotational alignment" begin
    testimage1 = zeros(Float64,10,10,10)
    testimage1 .= NaN
    testimage1[3:8,3:8,3:8] = gradcube
    testimage1 = centered(testimage1)

    tform = RotXYZ(0.1,0.1,0.1)
    mytform = AffineMap(tform, [0,0,0])
    testimage2 = warp(testimage1, mytform, axes(testimage1))

    # insufficient parameters + no initiat_tfm causes large misalignment

    tformtest1, mm1 = qd_rigid(testimage2, testimage1, mxshift, mxrot; print_interval=typemax(Int)) #
    @test !isapprox(1-mm1, 1)
    @test !isapprox(tformtest1, mytform, atol = 0.1)

    #initiat_tfm improves alignment

    tformtest2, mm2 = qd_rigid(testimage2, testimage1, mxshift, mxrot; print_interval=typemax(Int), initial_tfm = mytform) #
 #with the initial_tfm being the true tfm, this should give back the true rotation

    @test isapprox(1-mm2, 1)
    @test isapprox(tformtest2, mytform, atol = 0.0001)
    @test isrotation(tformtest2.linear)
end #Test initial_tfm improves rotational alignment for rigid

@testset "initial_tfm improves rigid translational alignment" begin
    testimage3 = zeros(10,20,10)
    testimage3 .= NaN
    testimage3[3:8,3:8,3:8] = gradcube
    testimage3 = centered(testimage3)

    testimage4 = zeros(10,20,10)
    testimage4 .= NaN
    testimage4[3:8,13:18,3:8] = gradcube
    testimage4 = centered(testimage4)

    init_tfm = AffineMap(diagm(0 => fill(1.0,3)), [0.0,10.0,0.0])

    tformtest3 = nothing
    mm3 = nothing

    #no initial_tfm should fail due to total lack of overlap
    @test try
        tformtest3, mm3 = qd_rigid(testimage3, testimage4, mxshift, mxrot; print_interval=typemax(Int)) #
        false
    catch err
        true  # this should break
    end

    tformtest4, mm4 = qd_rigid(testimage3, testimage4, mxshift, mxrot; print_interval=typemax(Int), initial_tfm = init_tfm) #

    @test isapprox(1-mm4, 1) #this mismatch should be lower!
    @test isapprox(tformtest4.translation, [0, 10, 0])
    @test isapprox(tformtest4.linear, EYE)
end


@testset "Test initial_tfm creates real rotations." begin #some of this seems redundant
    testimage1 = zeros(Float64,10,10,10)
    testimage1 .= NaN
    testimage1[3:8,3:8,3:8] = gradcube
    testimage1 = centered(testimage1)

    tform = RotXYZ(0.1,0.1,0.1)
    mytform = AffineMap(tform, [0,0,0])
    testimage2 = warp(testimage1, mytform, axes(testimage1))

    mxrot2 = (0.2,0.2,0.2);
    minwidth_rot2 = RegisterQD.default_minwidth_rot(CartesianIndices(testimage2), EYE) # TODO can we export this/ allow modification to, say 2* Default

    # tests with equal spaces produces real rotations
    tformtest0, mm0= qd_rigid(testimage2, testimage1, mxshift, mxrot2; print_interval=typemax(Int))

    @test isapprox(1-mm0, 1, atol = 0.0001)
    @test isapprox(tformtest0, mytform, atol = 0.1)
    @test isrotation(tformtest0.linear)

    tformtest01, mm01 = RegisterQD.qd_rigid_coarse(testimage2, testimage1, mxshift, [mxrot2...], minwidth_rot1; SD=EYE, print_interval=typemax(Int))
    @test isrotation(tformtest01.linear)

    tformtest02 = nothing
    mm02 = nothing
    tformtest02, mm02 = RegisterQD.qd_rigid_fine(testimage2, testimage1, [mxrot2...]./2, minwidth_rot1; SD=EYE, print_interval=typemax(Int))
    @test isrotation(tformtest02.linear)

    #with skewed spacing produces real rotations.
    testimage5 = testimage1[:,:,-4:2:5]
    testimage6 = testimage2[:,:,-4:2:5]
    ps = (1, 1, 2)
    SD =SArray{Tuple{3,3}}(Diagonal(SVector(ps) ./ minimum(ps)))

    tformtest5, mm5 = qd_rigid(testimage6, testimage5, mxshift, mxrot2; SD=SD, print_interval=typemax(Int))
    @test mm5 <1e-4
    @test isapprox(tformtest5, mytform, atol = 1)
    @test isrotation(tformtest5.linear)
    @test !isrotation(SD*tformtest5.linear*inv(SD))

    tformtest55, mm55 = qd_rigid(testimage6, testimage5, mxshift, mxrot2; SD=SD, print_interval=typemax(Int), initial_tfm = tformtest5)
    @test mm55 <1e-4
    @test isapprox(tformtest55, mytform, atol = 1)
    @test isrotation(tformtest55.linear)
    @test !isrotation(SD*tformtest55.linear*inv(SD))


    #coarse and find produce real rotations
    tformtest6, mm6 = RegisterQD.qd_rigid_coarse(testimage6, testimage5, mxshift, mxrot2, minwidth_rot2; SD=SD, print_interval=typemax(Int))
    @test mm6 <1e-4
    @test isrotation(tformtest6.linear)
    @test !isrotation(SD*tformtest6.linear*inv(SD))

    tformtest66, mm66 = RegisterQD.qd_rigid_coarse(testimage6, testimage5, mxshift, mxrot2, minwidth_rot2; SD=SD, print_interval=typemax(Int), initial_tfm = tformtest6)
    @test mm66 <1e-4
    @test isrotation(tformtest66.linear)
    @test !isrotation(SD*tformtest66.linear*inv(SD))

    tformtest7, mm7 = RegisterQD.qd_rigid_fine(testimage6, testimage5, [mxrot2...]./2, minwidth_rot2; SD=SD, print_interval=typemax(Int))
    @test mm7 <1e-4
    @test isrotation(tformtest7.linear)
    @test !isrotation(SD*tformtest7.linear*inv(SD))

    tformtest77, mm77 = RegisterQD.qd_rigid_fine(testimage6, testimage5, [mxrot2...]./2, minwidth_rot2; SD=SD, print_interval=typemax(Int), initial_tfm = tformtest7)
    @test mm77 <1e-4
    @test isrotation(tformtest77.linear)
    @test !isrotation(SD*tformtest77.linear*inv(SD))
    #fails due to specific error in qd_rigid_fine

    # a non rigid initial_tfm does not return a rigid transformation and prints an error message (?)
    tformtest8, mm8 = qd_rigid(testimage6, testimage5, mxshift, mxrot2; SD=SD, print_interval=typemax(Int), initial_tfm = AffineMap(SD\tformtest5.linear*SD, SD\tformtest5.translation))
    @test mm8 <1e-4
    @test isapprox(tformtest8, mytform, atol = 1)
    @test !isrotation(tformtest8.linear)

    mktemp() do path, io
        redirect_stdout(io) do
            tformtest8, mm8 = qd_rigid(testimage6, testimage5, mxshift, mxrot2; SD=SD, print_interval=typemax(Int), initial_tfm = AffineMap(SD\tformtest5.linear*SD, SD\tformtest5.translation))
        end
        flush(io)
        str = read(path, String)
        @test !isempty(str)
    end

end #TODO why do these pass when they're missing variables?

@testset "Test initial_tfm improves translational alignment for affine" begin
    testimage3 = zeros(10,20,10)
    testimage3 .= NaN
    testimage3[3:8,3:8,3:8] = gradcube
    testimage3 = centered(testimage3)

    testimage4 = zeros(10,20,10)
    testimage4 .= NaN
    testimage4[3:8,13:18,3:8] = gradcube
    testimage4 = centered(testimage4)

    init_tfm = AffineMap(diagm(0 => fill(1.0,3)), [0.0,10.0,0.0])

    tformtest3 = nothing
    mm3 = nothing

    @test try
        tformtest3, mm3 = qd_affine(testimage3, testimage4, mxshift; print_interval=typemax(Int)) #
        false
    catch err
        true  # this should break
    end

    tformtest4, mm4 = qd_affine(testimage3, testimage4, mxshift;  print_interval = typemax(Int), initial_tfm = init_tfm, fvalue = 1e-5) #
    @test isapprox(mm4, 0, atol = 0.0001) #this mismatch should be lower!
    @test isapprox(tformtest4.translation, [0.0, 10.0, 0.0])

end

@testset "Test initial_tfm improves general alignment for affine" begin
    testimage1 = zeros(Float64,10,10,10)
    testimage1 .= NaN
    testimage1[3:8,3:8,3:8] = gradcube
    testimage1 = centered(testimage1)

    tform = RotXYZ(0.1,0.1,0.1)
    mytform = AffineMap(tform, [0.0,0.0,0.0])
    testimage2 = warp(testimage1, mytform, axes(testimage1))


    tformtest1, mm1 = qd_affine(testimage2, testimage1, mxshift; print_interval=typemax(Int)) #
    #as the max-rotation is set too low, this should give a bad mismatch, but should still work
    @test !isapprox(1-mm1, 1)
    @test !isapprox(tformtest1, mytform, atol = 0.1)


    tformtest2, mm2 = qd_affine(testimage2, testimage1, mxshift; print_interval=typemax(Int), initial_tfm = mytform, fvalue = 1e-5) #
    #with the initial_tfm being the true tfm, this should give back the true rotation

    @test isapprox(1-mm2, 1)
    @test isapprox(tformtest2, mytform, atol = 0.0001) #this may not work so well. see qd_standard.
end #Test initial_tfm improves rotational alignment for rigid
