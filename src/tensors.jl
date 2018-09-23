struct SymTen{T<:Number} <: AbstractTen{T}
    xx::T
    xy::T
    xz::T
    yy::T 
    yz::T 
    zz::T
end

Base.IndexStyle(a::Type{<:SymTen}) = Base.IndexLinear()

@inline function Base.getindex(a::SymTen,I::Integer) 
    I <= 3 && return getfield(a,I)
    I == 4 && return a.xy
    I == 5 && return a.yy
    I == 6 && return a.yz
    I == 7 && return a.xz
    I == 8 && return a.yz
    I == 9 && return a.zz 
end

Base.size(a::SymTen) = 
    (3,3)

Base.length(a::SymTen) = 
    9

Base.zero(a::Type{SymTen{T}}) where {T} = 
    SymTen{T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T))

SymTen(xx::T,xy::T,xz::T,yy::T,yz::T,zz::T) where {T} = 
    SymTen{T}(xx,xy,xz,yy,yz,zz)

SymTen(q,w,e,r,t,y) = 
    SymTen(promote(q,w,e,r,t,y)...)

@inline Base.:+(a::SymTen{T},b::SymTen{T2}) where {T,T2} = 
    @fastmath SymTen{promote_type(T,T2)}(a.xx+b.xx, a.xy+b.xy, a.xz+b.xz, a.yy+b.yy, a.yz + b.yz, a.zz+b.zz)

@inline Base.:-(A::SymTen{T}) where T = SymTen{T}(-A.xx,-A.xy,-A.xz,-A.yy,-A.yz,-A.zz)

@inline Base.:-(a::SymTen{T},b::SymTen{T2}) where {T,T2} = 
    @fastmath SymTen{promote_type(T,T2)}(a.xx-b.xx, a.xy-b.xy, a.xz-b.xz, a.yy-b.yy, a.yz - b.yz, a.zz - b.zz)

@inline Base.:*(a::T,b::SymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath SymTen{promote_type(T,T2)}(a*b.xx, a*b.xy, a*b.xz, a*b.yy, a*b.yz, a*b.zz)

@inline Base.:*(b::SymTen{T},a::T2) where {T<:Number,T2<:Number} = 
    a*b

@inline Base.:/(b::SymTen{T},a::T2) where {T<:Number,T2<:Number} = 
    @fastmath inv(a)*b

@inline Base.:(:)(a::SymTen,b::SymTen) = 
    @fastmath muladd(a.xx, b.xx, muladd(a.yy, b.yy, a.zz*b.zz)) + 2muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*b.yz))

@inline Base.:(+)(L::LinearAlgebra.UniformScaling,A::SymTen) =
    SymTen(A.xx + L.λ, A.xy,A.xz,A.yy + L.λ, A.yz, A.zz + L.λ)

@inline Base.:(+)(A::SymTen, L::LinearAlgebra.UniformScaling) = L+A

@inline Base.:(-)(A::SymTen, L::LinearAlgebra.UniformScaling) =
    SymTen(A.xx - L.λ, A.xy,A.xz,A.yy - L.λ, A.yz, A.zz - L.λ)

@inline Base.:(-)(L::LinearAlgebra.UniformScaling,A::SymTen) = -(A-L)


@inline LinearAlgebra.tr(a::SymTen) =
    @fastmath a.xx + a.yy + a.zz

@inline LinearAlgebra.norm(a::SymTen) = 
    @fastmath sqrt(2(a:a))

@inline LinearAlgebra.dot(a::SymTen{T},b::Vec{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Vec{promote_type(T,T2)}(muladd(a.xx, b.x, muladd(a.xy, b.y, a.xz*b.z)), 
        muladd(a.xy, b.x, muladd(a.yy, b.y, a.yz*b.z)),
        muladd(a.xz, b.x, muladd(a.yz, b.y, a.zz*b.z)))

@inline LinearAlgebra.dot(b::Vec,a::SymTen) = 
    LinearAlgebra.dot(a,b)

@inline LinearAlgebra.dot(A::AbstractMatrix,v::Vec) = A*v

@inline symouter(a::Vec,b::Vec) = #symmetric part of the outer product of two vectors
    @fastmath SymTen(a.x*b.x,
                     0.5*muladd(a.x,b.y,a.y*b.x),
                     0.5*muladd(a.x,b.z,a.z*b.x),
                     a.y*b.y,
                     0.5*muladd(a.y,b.z,a.z*b.y),
                     a.z*b.z)

@inline Lie(S::SymTen,w::Vec) = #Lie product of Symmetric Tensor and Anti-symmetric tensor(passed as vector)
    @fastmath SymTen(2*(S.xz*w.y - S.xy*w.z),
                     muladd(w.y,S.yz, muladd(w.z,S.xx, -muladd(w.x,S.xz, w.z*S.yy))),
                     muladd(w.x,S.xy, muladd(w.y,S.zz, -muladd(w.y, S.xx, w.z*S.yz))),
                     2*(w.z*S.xy - w.x*S.yz),
                     muladd(w.x,S.yy, muladd(w.z,S.xz, -muladd(w.x,S.zz, w.y*S.xy))),
                     2*(w.x*S.yz - w.y*S.xz))
