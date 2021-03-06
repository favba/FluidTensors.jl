using Test
using FluidTensors
using LinearAlgebra

@testset "Vector Operations" begin
    a = Vec(1.,2.,3.)
    b = Vec(4.,5.,6.)

    @test a == [1.,2.,3.]
    @test a+b === Vec(5.,7.,9.)
    @test b-a === Vec(3.,3.,3.)
    @test a⋅b ≈ 32.
    @test norm(a) ≈ 3.7416573867739413
    @test norm(im*a) ≈ 3.7416573867739413
    @test 2*a === Vec(2.,4.,6.)
    @test a/2 === Vec(0.5,1.,1.5)
    @test cross(a,b) === Vec(-3.,6.,-3.)
    @test cross(b,a) === Vec(3.,-6.,3.)
end

@testset "Tensor Operations" begin
    A = SymTen(1.,2.,3.,4.,5.,6.)
    @test A == [1. 2. 3.;
                2. 4. 5.;
                3. 5. 6.]
    B = SymTen(2.,3.,4.,5.,6.,7.)
    @test A+B === SymTen(3.,5.,7.,9.,11.,13.)
    @test B-A === SymTen(1.,1.,1.,1.,1.,1.)
    @test 2*A === SymTen(2.,4.,6.,8.,10.,12.)
    @test A/2 === SymTen(0.5,1.,1.5,2.,2.5,3.)
    @test tr(A) == 11.
    @test A:B == tr(A*B)
    @test norm(A) == sqrt(2*tr(A*A))
    @test traceless(SymTen(1.,2.,3.,1.,5.,1.)) == [0. 2. 3.;
                                                   2. 0. 5.;
                                                   3. 5. 0.]
end

@testset "Tensor x Vector operation" begin
    a = Vec(1.,2.,3.)
    b = Vec(4.,5.,6.)
    A = SymTen(1.,2.,3.,4.,5.,6.)
    @test A⋅a === Vec(14.,25.,31.)
    @test symouter(a,b) === SymTen(4.,6.5,9.,10.,13.5,18.)
    @test antisymouter(a,b) === AntiSymTen(-1.5,-3.0,-1.5)
    @test outer(a,b) === Ten(4.,8.,12.,5.,10.,15.,6.,12.,18.)
end

@testset "UniformScaling" begin
    A = SymTen(1,1,1,1,1,1)
    @test A - I === SymTen(0,1,1,0,1,0)
    @test A + I === SymTen(2,1,1,2,1,2)
    @test I - A === SymTen(0,-1,-1,0,-1,0)
end

@testset "VecArrays" begin
    v = VecArray([1. 2.;3. 4.], [5. 6.; 7. 8.], [9. 10.;-1. -2.])
    @test eltype(v) === Vec{Float64}
    @test size(v) == (2,2)
    @test v[2] === Vec{Float64}(3.,7.,-1.)
    @test v[1,2] === Vec{Float64}(2., 6., 10.)
    @test v[2,2] === Vec{Float64}(4., 8., -2.)
    v[1] = Vec(-1.,-2.,-3.)
    @test v.x[1] == -1 && v.y[1] == -2 && v.z[1] == -3.
end

@testset "Ten Mixed type operations" begin
    v = Vec(rand(3)...)
    for a in (SymTen(rand(6)...), AntiSymTen(rand(3)...), Ten(rand(9)...))
        for b in (SymTen(rand(6)...), AntiSymTen(rand(3)...), Ten(rand(9)...))
            for op in (+,-)
                @test op(a,b) ≈ op(Matrix(a),Matrix(b))
            end
            @test a⋅b ≈ a*b
            @test a:b ≈ tr(Matrix(a)'*Matrix(b)) atol=2e-15
            @test Lie(a,b) ≈ (Matrix(a)*Matrix(b) - Matrix(b)*Matrix(a)) atol=2e-15
        end
        @test a⋅v ≈ Matrix(a)*Vector(v)
        @test v⋅a ≈ Matrix(a)'*Vector(v)
        @test -a ≈ -Matrix(a)
        @test a + I ≈ Matrix(a) + I
        @test a - I ≈ Matrix(a) - I
        @test I + a ≈ I + Matrix(a)
        @test I - a ≈ I - Matrix(a)

        for op in (+,-,tr,det)
            @test op(a) ≈ op(Matrix(a)) atol=2e-15
        end

        @test square(a) ≈ Matrix(a)^2

        @test symmetric(a) ≈ (Matrix(a) + Matrix(a)')/2 atol=2e-15
        @test antisymmetric(a) ≈ (Matrix(a) - Matrix(a)')/2 atol=2e-15

    end
end

@testset "Eigenvalues and vectors of SymTen" begin
    for a in (SymTen(rand(6)...), SymTen(rand(6)...), SymTen(rand(6)...))
        l,e = eigvec(a)
        @test all(isapprox.(l, eig(a))) || all(isapprox.((l[3],l[2],l[1]), eig(a)))
        b = eigen(Matrix(a))
        lv = b.values
        ev = b.vectors
        @test all(isapprox.(l, lv))
        @test (e[1] ≈ ev[:,1] || -e[1] ≈ ev[:,1])
        @test (e[2] ≈ ev[:,2] || -e[2] ≈ ev[:,2])
        @test (e[3] ≈ ev[:,3] || -e[3] ≈ ev[:,3])

        @test cross(e[1],e[2]) ≈ e[3]
    end
end