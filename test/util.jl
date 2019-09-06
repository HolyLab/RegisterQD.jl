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

#TODO add a testset for other support functions
#rotations
