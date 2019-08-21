using StaticArrays, Interpolations, LinearAlgebra
using Images, CoordinateTransformations, Rotations
using RegisterQD

gradcube = zeros(5,5,5)
gradcube[:,:,1] = [0.2, 0.4, 0.6, 0.8, 1.0]*[0.2 0.4 0.6 0.8 1.0]
for i = 1:5
    gradcube[i,:,:] = gradcube[i,:,1]*[0.2 0.4 0.6 0.8 1.0]
end
gradcube[1,4,2] = 0
gradcube[4,5,2] = 0 #break up any potential symmetry

testimage3 = zeros(10,20,10)
testimage3 .= NaN
testimage3[4:8,4:8,4:8] = gradcube
testimage3 = centered(testimage3)

testimage4 = zeros(10,20,10)
testimage4 .= NaN
testimage4[4:8,14:18,4:8] = gradcube
testimage4 = centered(testimage4)

mxshift = (1,1,1);
mxrot = (0.01,0.01,0.01);
minwidth_rot = fill(0.002,3);

init_tfm = AffineMap(diagm(0 => fill(1,3)), [0,10,0])

try
    tformtest2, mm2 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int)) #
catch err # this should break
end

try
    tformtest3, mm3 = qd_rigid(testimage3, testimage4, mxshift, mxrot, minwidth_rot; print_interval=typemax(Int), initial_tfm = init_tfm) #
catch err # this should work
end

@test !(@isdefined tformtest2)
@test @isdefined tformtest3
@test isapprox(1-mm3, 1)
