struct HomogeneousArray{T,N,A<:Union{<:AbstractVector,<:NTuple},P} <: AbstractArray{T,N}
    data::A
    size::NTuple{N,Int}
    HomogeneousArray{T,N,A,P}(data::A,size::NTuple{N,Int}) where {T,N,A,P} = (@assert 1 <= P <= N; new{T,N,A,P}(data,size))
end

@inline HomogeneousArray{P}(data,size) where P = HomogeneousArray{eltype(data),length(size),typeof(data),P}(data,size)

Base.IndexStyle(::Type{<:HomogeneousArray}) = IndexCartesian()

@inline Base.size(a::HomogeneousArray) = a.size

@inline Base.@propagate_inbounds function Base.getindex(v::HomogeneousArray{T,N,A,P},I::Vararg{Int,N}) where {T,N,A,P}
    d = v.data
    i = @inbounds I[P]
    return d[I[P]]
end