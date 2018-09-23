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

LinearAlgebra.norm(a::Vec{<:Real}) = @fastmath sqrt(muladd(xpos(a), xpos(a), muladd(ypos(a), ypos(a), zpos(a)^2)))
LinearAlgebra.norm(a::Vec) = @fastmath sqrt(abs2(a.x)+abs2(a.y)+abs2(a.z))

distance(a::AbstractVec,b::AbstractVec) = @fastmath sqrt((xpos(b)-xpos(a))^2 + (ypos(b)-ypos(a))^2 + (zpos(b)-zpos(a))^2)

@inline Base.:+(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)+xpos(b), ypos(a)+ypos(b), zpos(a)+zpos(b))
@inline Base.:-(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)-xpos(b), ypos(a)-ypos(b), zpos(a)-zpos(b))
@inline Base.:*(a::Number,b::AbstractVec) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
@inline Base.:*(b::AbstractVec,a::Number) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
@inline Base.:/(b::AbstractVec,a::Number) = @fastmath inv(a)*b

@inline LinearAlgebra.dot(a::AbstractVec,b::AbstractVec) = @fastmath muladd(xpos(a), xpos(b), muladd(ypos(a), ypos(b), zpos(a)*zpos(b)))

@inline LinearAlgebra.cross(a::AbstractVec,b::AbstractVec) = @fastmath Vec(ypos(a)*zpos(b) - zpos(a)*ypos(b), zpos(a)*xpos(b) - xpos(a)*zpos(b), xpos(a)*ypos(b) - ypos(a)*xpos(b))


@inline Base.size(a::Vec) = (3,)
@inline Base.length(a::Vec) = 3
Base.zero(a::Type{Vec{T}}) where {T} = Vec{T}(zero(T),zero(T),zero(T))
Vec(x::T,y::T,z::T) where T = Vec{T}(x,y,z)
Vec(x,y,z) = Vec(promote(x,y,z)...)
