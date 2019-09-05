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
#####################################################################################################
#Test initial_tfm improves rotational alignment
#####################################################################################################

testimage1 = zeros(Float64,10,10,10)
testimage1 .= NaN
testimage1[3:8,3:8,3:8] = gradcube
testimage1 = centered(testimage1)

tform = RotXYZ(0.1,0.1,0.1)
mytform = AffineMap(tform, [0,0,0])
testimage2 = warp(testimage1, mytform, axes(testimage1))

tformtest1 = nothing
tformtest2 = nothing
mm1 = nothing
mm2 = nothing

@test try
    global tformtest1, mm1 = qd_rigid(testimage2, testimage1, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int)) #
    true
catch err
    false
end #as the max-rotation is set too low, this should give a bad mismatch
# this set is kind of pointless because a wrong rotation doesn't cause it to fail

@test try
    global tformtest2, mm2 = qd_rigid(testimage2, testimage1, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int), initial_tfm = mytform) #
    true
catch err
    false
end #with the initial_tfm being the true tfm, this should give back the true rotation

@test !isapprox(1-mm1, 1)
@test !isapprox(tformtest1, mytform, atol = 0.1)

@test isapprox(1-mm2, 1)
@test isapprox(tformtest2, mytform, atol = 0.0001)


#####################################################################################################
#Test initial_tfm improves translational alignment
#####################################################################################################

testimage3 = zeros(10,20,10)
testimage3 .= NaN
testimage3[3:8,3:8,3:8] = gradcube
testimage3 = centered(testimage3)

testimage4 = zeros(10,20,10)
testimage4 .= NaN
testimage4[3:8,13:18,3:8] = gradcube
testimage4 = centered(testimage4)

init_tfm = AffineMap(diagm(0 => fill(1,3)), [0,10,0])

tformtest3 = nothing
tformtest4 = nothing
mm3 = nothing
mm4 = nothing

@test try
    global tformtest3, mm3 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int)) #
    false
catch err
    true  # this should break
end

@test try
    global tformtest4, mm4 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int), initial_tfm = init_tfm) #
    true
catch err # this should work
    false
end

@test isapprox(1-mm4, 1) #this mismatch should be lower!
@test isapprox(tformtest4.translation, [0, 10, 0])
@test isapprox(tformtest4.linear, EYE)

#####################################################################################################
#Test initial_tfm creates real rotations.
#####################################################################################################

mxrot2 = (0.2,0.2,0.2);



# With no SD
tformtest0, mm0= qd_rigid(testimage2, testimage1, mxshift, mxrot2, minwidth_rot; print_interval=typemax(Int))
@test isapprox(1-mm0, 1, atol = 0.0001)
@test isapprox(tformtest0, mytform, atol = 0.1)
@test isrotation(tformtest0.linear)

tformtest01, mm01 = RegisterQD.qd_rigid_coarse(testimage2, testimage1, mxshift, [mxrot2...], minwidth_rot, EYE; print_interval=typemax(Int))
@test isrotation(tformtest01.linear)

tformtest02, mm02 = RegisterQD.qd_rigid_fine(testimage2, testimage1, [mxrot2...]./2, minwidth_rot, EYE; print_interval=typemax(Int))
@test isrotation(tformtest02.linear)

#with SD
testimage5 = testimage1[:,:,-4:2:5]
testimage6 = testimage2[:,:,-4:2:5]
ps = (1, 1, 2)
SD =SArray{Tuple{3,3}}(Diagonal(SVector(ps) ./ minimum(ps)))

tformtest5, mm5 = qd_rigid(testimage6, testimage5, mxshift, mxrot2, minwidth_rot, SD; print_interval=typemax(Int))
@test isapprox(1-mm5, 1, atol = 0.0001)
@test isapprox(tformtest5, mytform, atol = 1)
@test !isrotation(tformtest5.linear)
@test isrotation(SD*tformtest5.linear*inv(SD))

tformtest50, mm50 = qd_rigid(testimage6, testimage5, mxshift, mxrot2, minwidth_rot, SD; print_interval=typemax(Int), initial_tfm = AffineMap(inv(SD)*mytform.linear*SD, mytform.translation))
@test isapprox(1-mm50, 1, atol = 0.001)
@test isapprox(tformtest50, mytform, atol = 1)
@test !isrotation(tformtest50.linear)
@test_broken isrotation(SD*tformtest50.linear*inv(SD))

tformtest55, mm55 = qd_rigid(testimage6, testimage5, mxshift, mxrot2, minwidth_rot, SD; print_interval=typemax(Int), initial_tfm = tformtest5)
@test isapprox(1-mm55, 1, atol = 0.0001)
@test isapprox(tformtest55, mytform, atol = 1)
@test !isrotation(tformtest55.linear)
@test isrotation(SD*tformtest55.linear*inv(SD))
#I should probably run this with debugger to see why this works

tformtest6, mm6 = RegisterQD.qd_rigid_coarse(testimage6, testimage5, mxshift, mxrot2, minwidth_rot, SD; print_interval=typemax(Int))
@test !isrotation(tformtest6.linear)
@test isrotation(SD*tformtest6.linear*inv(SD))

tformtest66, mm66 = RegisterQD.qd_rigid_coarse(testimage6, testimage5, mxshift, mxrot2, minwidth_rot, SD; print_interval=typemax(Int), initial_tfm = tformtest6)
@test !isrotation(tformtest66.linear)
@test isrotation(SD*tformtest66.linear*inv(SD))

tformtest7, mm7 = RegisterQD.qd_rigid_fine(testimage6, testimage5, [mxrot2...]./2, minwidth_rot, SD; print_interval=typemax(Int))
@test !isrotation(tformtest7.linear)
@test isrotation(SD*tformtest7.linear*inv(SD))

tformtest77, mm77 = RegisterQD.qd_rigid_fine(testimage6, testimage5, [mxrot2...]./2, minwidth_rot, SD; print_interval=typemax(Int), initial_tfm = tformtest7)
@test !isrotation(tformtest77.linear)
@test isrotation(SD*tformtest77.linear*inv(SD))
#fails due to specific error in qd_rigid_fine
