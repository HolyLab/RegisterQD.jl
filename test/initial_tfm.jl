using StaticArrays, Interpolations, LinearAlgebra
using Images, CoordinateTransformations, Rotations
using RegisterQD

gradcube = zeros(5,5,5)
gradcube[:,:,1] = [0.2, 0.4, 0.6, 0.8, 1.0]*[0.2 0.4 0.6 0.8 1.0]
for i = 1:5
    gradcube[i,:,:] = gradcube[i,:,1]*[0.2 0.4 0.6 0.8 1.0]
end
gradcube[1,4,2] = 0
gradcube[4,5,2] = 0 #break up any potential rotational symmetry

mxshift = (1,1,1);
mxrot = (0.01,0.01,0.01);
minwidth_rot = fill(0.002,3);


#####################################################################################################
#Test initial_tfm improves rotational alignment
#####################################################################################################

testimage1 = zeros(Float64,10,10,10)
testimage1 .= NaN
testimage1[4:8,4:8,4:8] = gradcube
testimage1 = centered(testimage1)

tform = RotXYZ(0.1,0.1,0.1)
mytform = AffineMap(tform, [0,0,0])
testimage2 = warp(testimage1, mytform, axes(testimage1))

tformtest1 = nothing
tformtest2 = nothing
mm1 = nothing
mm2 = nothing

try
    global tformtest1, mm1 = qd_rigid(testimage2, testimage1, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int)) #
catch err
end #as the max-rotation is set too low, this should give a bad mismatch

try
    global tformtest2, mm2 = qd_rigid(testimage2, testimage1, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int), initial_tfm = mytform) #
catch err
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
testimage3[4:8,4:8,4:8] = gradcube
testimage3 = centered(testimage3)

testimage4 = zeros(10,20,10)
testimage4 .= NaN
testimage4[4:8,14:18,4:8] = gradcube
testimage4 = centered(testimage4)

init_tfm = AffineMap(diagm(0 => fill(1,3)), [0,10,0])

tformtest3 = nothing
tformtest4 = nothing
mm3 = nothing
mm4 = nothing

try
    global tformtest3, mm3 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int)) #
catch err # this should break
end

try
    global tformtest4, mm4 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int), initial_tfm = init_tfm) #
catch err # this should work
end

@test isnothing(tformtest3)
@test @isdefined tformtest4
@test isapprox(1-mm4, 1)
