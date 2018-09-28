struct HomogeneousArray{T,N,A<:Union{<:AbstractVector,<:NTuple},P} <: AbstractArray{T,N}
    data::A
    size::NTuple{N,Int}
end

HomogeneousArray{P}(data,size) where P = HomogeneousArray{eltype(data),length(size),typeof(data),P}(data,size)

Base.IndexStyle(::Type{<:HomogeneousArray}) = IndexCartesian()

Base.size(a::HomogeneousArray) = a.size

@inline Base.@propagate_inbounds function Base.getindex(v::HomogeneousArray{T,N,A,P},I::Vararg{Int,N}) where {T,N,A,P}
    d = v.data
    return d[I[P]]
end