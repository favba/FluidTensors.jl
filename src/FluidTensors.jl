module FluidTensors

import LinearAlgebra

export Vec, SymTen, symouter, Lie, traceless, AbstractVecArray, VecArray, AbstractSymTenArray, AbstractSymTrTenArray, SymTenArray, SymTrTenArray

abstract type AbstractVec{T} <: AbstractArray{T,1} end
abstract type AbstractTen{T} <: AbstractArray{T,2} end

include("vectors.jl")
include("tensors.jl")

end
