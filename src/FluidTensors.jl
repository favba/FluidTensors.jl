module FluidTensors

import LinearAlgebra
using LinearAlgebra

export Vec, SymTen, symouter, Lie, traceless, AbstractVecArray, VecArray, AbstractSymTenArray, AbstractSymTrTenArray, SymTenArray, SymTrTenArray, AbstractAntiSymTenArray, AntiSymTenArray, HomogeneousArray, eig, eigvec, anglecos, stress_state
export prop_decomp, inph_decomp, Ten, outer, ⊗, AntiSymTen, square, symmetric, antisymmetric

abstract type AbstractVec{T} <: AbstractArray{T,1} end
abstract type AbstractTen{T} <: AbstractArray{T,2} end

include("vectors.jl")
include("tensors.jl")
include("utils.jl")
include("eigen.jl")
include("decompositions.jl")

end
