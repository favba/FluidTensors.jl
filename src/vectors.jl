@inline Base.@propagate_inbounds Base.getindex(a::AbstractVec,I::Integer) = getfield(a,I)
Base.IndexStyle(a::Type{<:AbstractVec}) = Base.IndexLinear()

@inline xpos(a::AbstractVec) = a.x
@inline ypos(a::AbstractVec) = a.y
@inline zpos(a::AbstractVec) = a.z

#pos of a node returns a tuple with the positions
pos(a::AbstractVec) = (xpos(a),ypos(a),zpos(a))

LinearAlgebra.norm(a::AbstractVec) = @fastmath sqrt(muladd(xpos(a), xpos(a), muladd(ypos(a), ypos(a), zpos(a)^2)))
distance(a::AbstractVec,b::AbstractVec) = @fastmath sqrt((xpos(b)-xpos(a))^2 + (ypos(b)-ypos(a))^2 + (zpos(b)-zpos(a))^2)

Base.:+(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)+xpos(b), ypos(a)+ypos(b), zpos(a)+zpos(b))
Base.:-(a::AbstractVec,b::AbstractVec) = @fastmath Vec(xpos(a)-xpos(b), ypos(a)-ypos(b), zpos(a)-zpos(b))
Base.:*(a::Number,b::AbstractVec) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
Base.:*(b::AbstractVec,a::Number) = @fastmath Vec(a*xpos(b), a*ypos(b), a*zpos(b))
Base.:/(b::AbstractVec,a::Number) = @fastmath inv(a)*b

LinearAlgebra.dot(a::AbstractVec,b::AbstractVec) = @fastmath muladd(xpos(a), xpos(b), muladd(ypos(a), ypos(b), zpos(a)*zpos(b)))

LinearAlgebra.cross(a::AbstractVec,b::AbstractVec) = @fastmath Vec(ypos(a)*zpos(b) - zpos(a)*ypos(b), zpos(a)*xpos(b) - xpos(a)*zpos(b), xpos(a)*ypos(b) - ypos(a)*xpos(b))

struct Vec{T<:Number} <: AbstractVec{T}
    x::T
    y::T
    z::T
end

Base.size(a::Vec) = (3,)
Base.length(a::Vec) = 3
Base.zero(a::Type{Vec{T}}) where {T} = Vec{T}(zero(T),zero(T),zero(T))
Vec(x::T,y::T,z::T) where T = Vec{T}(x,y,z)
Vec(x,y,z) = Vec(promote(x,y,z)...)
