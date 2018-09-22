module FluidTensors

import LinearAlgebra

export Vec, SymTen, symouter, Lie

abstract type AbstractVec{T} <: DenseArray{T,1} end
abstract type AbstractTen{T} <: AbstractArray{T,2} end

include("vectors.jl")
include("tensors.jl")

end
