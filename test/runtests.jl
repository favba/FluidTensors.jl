using Test
using FluidTensors
using LinearAlgebra

@testset "Vector Operations" begin
    a = Vec(1.,2.,3.)
    b = Vec(4.,5.,6.)

    @test a == [1.,2.,3.]
    @test a+b == Vec(5.,7.,9.)
    @test b-a == Vec(3.,3.,3.)
    @test a⋅b ≈ 32.
    @test norm(a) ≈ 3.7416573867739413
    @test norm(im*a) ≈ 3.7416573867739413
    @test 2*a == Vec(2.,4.,6.)
    @test a/2 == Vec(0.5,1.,1.5)
    @test cross(a,b) == Vec(-3.,6.,-3.)
    @test cross(b,a) == Vec(3.,-6.,3.)
end

@testset "Tensor Operations" begin
    A = SymTen(1.,2.,3.,4.,5.,6.)
    @test A == [1. 2. 3.;
                2. 4. 5.;
                3. 5. 6.]
    B = SymTen(2.,3.,4.,5.,6.,7.)
    @test A+B == SymTen(3.,5.,7.,9.,11.,13.)
    @test B-A == SymTen(1.,1.,1.,1.,1.,1.)
    @test 2*A == SymTen(2.,4.,6.,8.,10.,12.)
    @test A/2 == SymTen(0.5,1.,1.5,2.,2.5,3.)
    @test tr(A) == 11.
    @test A:B == tr(A*B)
    @test norm(A) == sqrt(2*tr(A*A))
end

@testset "Tensor x Vector operation" begin
    a = Vec(1.,2.,3.)
    b = Vec(4.,5.,6.)
    A = SymTen(1.,2.,3.,4.,5.,6.)
    @test A⋅a == Vec(14.,25.,31.)
    W = [0. a.z -a.y;
         -a.z 0. a.x;
         a.y -a.x 0.]
    @test Lie(A,a) ==  A*W - W*A
    @test symouter(a,b) == SymTen(4.,6.5,9.,10.,13.5,18.)
end