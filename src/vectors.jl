struct Vec{T<:Number} <: AbstractVec{T}
    x::T
    y::T
    z::T
end

@inline Base.@propagate_inbounds Base.getindex(a::AbstractVec,I::Integer) = getfield(a,I)
Base.IndexStyle(a::Type{<:AbstractVec}) = Base.IndexLinear()

@inline xpos(a::AbstractVec) = a.x
@inline ypos(a::AbstractVec) = a.y
@inline zpos(a::AbstractVec) = a.z

@inline LinearAlgebra.norm(a::Vec{<:Real}) = @fastmath sqrt(muladd(xpos(a), xpos(a), muladd(ypos(a), ypos(a), zpos(a)^2)))
@inline LinearAlgebra.norm(a::Vec) = @fastmath sqrt(abs2(a.x)+abs2(a.y)+abs2(a.z))

@inline Base.:+(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)+xpos(b), ypos(a)+ypos(b), zpos(a)+zpos(b))
@inline Base.:-(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)-xpos(b), ypos(a)-ypos(b), zpos(a)-zpos(b))
@inline Base.:*(a::Number,b::AbstractVec) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
@inline Base.:*(b::AbstractVec,a::Number) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
@inline Base.:/(b::AbstractVec,a::Number) = @fastmath inv(a)*b

@inline LinearAlgebra.dot(a::AbstractVec,b::AbstractVec) = @fastmath muladd(xpos(a), xpos(b), muladd(ypos(a), ypos(b), zpos(a)*zpos(b)))

@inline LinearAlgebra.cross(a::AbstractVec,b::AbstractVec) = @fastmath Vec(ypos(a)*zpos(b) - zpos(a)*ypos(b), zpos(a)*xpos(b) - xpos(a)*zpos(b), xpos(a)*ypos(b) - ypos(a)*xpos(b))

@inline anglecos(a::Vec,b::Vec) = LinearAlgebra.dot(a,b)/(LinearAlgebra.norm(a)*LinearAlgebra.norm(b))

@inline Base.size(a::Vec) = (3,)
@inline Base.length(a::Vec) = 3
@inline Base.zero(a::Type{Vec{T}}) where {T} = Vec{T}(zero(T),zero(T),zero(T))
@inline Vec(x::T,y::T,z::T) where T = Vec{T}(x,y,z)
@inline Vec(x,y,z) = Vec(promote(x,y,z)...)


abstract type AbstractVecArray{T,N} <: AbstractArray{Vec{T},N} end

@inline Base.@propagate_inbounds function Base.getindex(v::AbstractVecArray{T,N},i::Int) where {T,N}
    x = xvec(v)
    y = yvec(v)
    z = zvec(v)
    xv = x[i]
    yv = y[i]
    zv = z[i]
    return Vec{T}(xv,yv,zv)
end

@inline Base.@propagate_inbounds function Base.getindex(v::AbstractVecArray{T,N},I::Vararg{Int,N}) where {T,N}
    x = xvec(v)
    y = yvec(v)
    z = zvec(v)
    xv = x[I...]
    yv = y[I...]
    zv = z[I...]
    return Vec{T}(xv,yv,zv)
end

@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractVecArray,vec,i::Int)
    x = xvec(v)
    y = yvec(v)
    z = zvec(v)
    setindex!(x,getfield(vec,1),i)
    setindex!(y,getfield(vec,2),i)
    setindex!(z,getfield(vec,3),i)
    return v
end

@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractVecArray{T,N},vec,I::Vararg{Int,N}) where {T,N}
    x = xvec(v)
    y = yvec(v)
    z = zvec(v)
    setindex!(x,getfield(vec,1),I...)
    setindex!(y,getfield(vec,2),I...)
    setindex!(z,getfield(vec,3),I...)
    return v
end


@inline Base.size(v::AbstractVecArray) =
    size(xvec(v))

struct VecArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N},C<:AbstractArray{T,N}} <: AbstractVecArray{T,N}
    x::A
    y::B
    z::C
    function VecArray{T,N,A,B,C}(x::A,y::B,z::C) where {T,N,A,B,C}
        @assert (size(x) == size(y) == size(z))
        return new{T,N,A,B,C}(x,y,z)
    end
end

@inline VecArray(x::AbstractArray{T,N},y::B,z::C) where{T,N,B<:AbstractArray{T,N},C<:AbstractArray{T,N}} = VecArray{T,N,typeof(x),B,C}(x,y,z)
@inline VecArray{T}(dims::Vararg{Int,N}) where {T,N} = VecArray(zeros(T,dims...),zeros(T,dims...),zeros(T,dims...))
@inline VecArray(dims::Vararg{Int,N}) where {N} = VecArray{Float64}(dims...)

Base.read!(io::NTuple{3,A},a::VecArray) where{A<:Union{<:IO,<:AbstractString}} = (read!(io[1],a.x); read!(io[2],a.y); read!(io[3],a.z))
Base.write(io::NTuple{3,A},a::VecArray) where{A<:Union{<:IO,<:AbstractString}} = (write(io[1],a.x); write(io[2],a.y); write(io[3],a.z))

@inline xvec(v::VecArray) =
    v.x
@inline yvec(v::VecArray) =
    v.y
@inline zvec(v::VecArray) =
    v.z

Base.similar(a::VecArray) = VecArray(similar(a.x),similar(a.y),similar(a.z))