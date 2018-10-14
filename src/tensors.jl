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

@inline Base.size(a::SymTen) = 
    (3,3)

@inline Base.length(a::SymTen) = 
    9

@inline Base.zero(a::Type{SymTen{T}}) where {T} = 
    SymTen{T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T))

@inline SymTen(xx::T,xy::T,xz::T,yy::T,yz::T,zz::T) where {T} = 
    SymTen{T}(xx,xy,xz,yy,yz,zz)

@inline SymTen(q,w,e,r,t,y) = 
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

@inline traceless(S::SymTen) =
    S - inv(3)*LinearAlgebra.tr(S)*LinearAlgebra.I

@inline function Base.:^(t::SymTen{T},n::Integer) where {T}
    λ,e = eigvec(t)
    r = (λ[1]^n)*(symouter(e[1],e[1])) + (λ[2]^n)*(symouter(e[2],e[2])) + (λ[3]^n)*(symouter(e[3],e[3]))
end

@inline function Base.literal_pow(::typeof(Base.:^),t::SymTen{T},::Val{2}) where {T<:AbstractFloat}
    @fastmath SymTen{T}(muladd(t.xx,t.xx,muladd(t.xy,t.xy,t.xz^2)),
              muladd(t.xx,t.xy,muladd(t.yy,t.xy,t.xz*t.yz)),
              muladd(t.xx,t.xz,muladd(t.xy,t.yz,t.zz*t.xz)),
              muladd(t.xy,t.xy,muladd(t.yy,t.yy,t.yz^2)),
              muladd(t.xy,t.xz,muladd(t.yy,t.yz,t.zz*t.yz)),
              muladd(t.xz,t.xz,muladd(t.yz,t.yz,t.zz^2)))
end

############################### Array types ########################


abstract type AbstractSymTenArray{T,N} <: AbstractArray{SymTen{T},N} end

Base.size(v::AbstractSymTenArray) =
    size(xxvec(v))

@inline Base.@propagate_inbounds function Base.getindex(v::AbstractSymTenArray{T,N},i::Int) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    zz = zzvec(v)
    xxv = xx[i]
    xyv = xy[i]
    xzv = xz[i]
    yyv = yy[i]
    yzv = yz[i]
    zzv = zz[i]
    return SymTen{T}(xxv,xyv,xzv,yyv,yzv,zzv)
end
    
@inline Base.@propagate_inbounds function Base.getindex(v::AbstractSymTenArray{T,N},I::Vararg{Int,N}) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    zz = zzvec(v)
    xxv = xx[I...]
    xyv = xy[I...]
    xzv = xz[I...]
    yyv = yy[I...]
    yzv = yz[I...]
    zzv = zz[I...]
    return SymTen{T}(xxv,xyv,xzv,yyv,yzv,zzv)
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractSymTenArray,ten,i::Int)
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    zz = zzvec(v)
    setindex!(xx,ten.xx,i)
    setindex!(xy,ten.xy,i)
    setindex!(xz,ten.xz,i)
    setindex!(yy,ten.yy,i)
    setindex!(yz,ten.yz,i)
    setindex!(zz,ten.zz,i)
    return v
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractSymTenArray{T,N},ten,I::Vararg{Int,N}) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    zz = zzvec(v)
    setindex!(xx,ten.xx,I...)
    setindex!(xy,ten.xy,I...)
    setindex!(xz,ten.xz,I...)
    setindex!(yy,ten.yy,I...)
    setindex!(yz,ten.yz,I...)
    setindex!(zz,ten.zz,I...)
    return v
end

struct SymTenArray{T,N,A<:AbstractArray{T,N}} <: AbstractSymTenArray{T,N}
    xx::A
    xy::A
    xz::A
    yy::A
    yz::A
    zz::A
end

@inline xxvec(v::SymTenArray) =
    v.xx
@inline xyvec(v::SymTenArray) =
    v.xy
@inline xzvec(v::SymTenArray) =
    v.xz
@inline yyvec(v::SymTenArray) =
    v.yy
@inline yzvec(v::SymTenArray) =
    v.yz
@inline zzvec(v::SymTenArray) =
    v.zz

Base.similar(a::SymTenArray) = SymTenArray(similar(a.xx),similar(a.xy),similar(a.xz),similar(a.yy),similar(a.yz),similar(a.zz))

abstract type AbstractSymTrTenArray{T,N} <: AbstractArray{SymTen{T},N} end

@inline Base.@propagate_inbounds function Base.getindex(v::AbstractSymTrTenArray{T,N},i::Int) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    xxv = xx[i]
    xyv = xy[i]
    xzv = xz[i]
    yyv = yy[i]
    yzv = yz[i]
    return SymTen{T}(xxv,xyv,xzv,yyv,yzv,-(xxv+yyv))
end
    
@inline Base.@propagate_inbounds function Base.getindex(v::AbstractSymTrTenArray{T,N},I::Vararg{Int,N}) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    xxv = xx[I...]
    xyv = xy[I...]
    xzv = xz[I...]
    yyv = yy[I...]
    yzv = yz[I...]
    return SymTen{T}(xxv,xyv,xzv,yyv,yzv,-(xxv+yyv))
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractSymTrTenArray,ten,i::Int)
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    ten = traceless(ten)
    setindex!(xx,ten.xx,i)
    setindex!(xy,ten.xy,i)
    setindex!(xz,ten.xz,i)
    setindex!(yy,ten.yy,i)
    setindex!(yz,ten.yz,i)
    return v
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractSymTrTenArray{T,N},vec,I::Vararg{Int,N}) where {T,N}
    xx = xxvec(v)
    xy = xyvec(v)
    xz = xzvec(v)
    yy = yyvec(v)
    yz = yzvec(v)
    ten = traceless(vec)
    setindex!(xx,ten.xx,I...)
    setindex!(xy,ten.xy,I...)
    setindex!(xz,ten.xz,I...)
    setindex!(yy,ten.yy,I...)
    setindex!(yz,ten.yz,I...)
    return v
end

Base.size(v::AbstractSymTrTenArray) =
    size(xxvec(v))

struct SymTrTenArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N},C<:AbstractArray{T,N},D<:AbstractArray{T,N},E<:AbstractArray{T,N}} <: AbstractSymTrTenArray{T,N}
    xx::A
    xy::B
    xz::C
    yy::D
    yz::E
end
    
@inline xxvec(v::SymTrTenArray) =
    v.xx
@inline xyvec(v::SymTrTenArray) =
    v.xy
@inline xzvec(v::SymTrTenArray) =
    v.xz
@inline yyvec(v::SymTrTenArray) =
    v.yy
@inline yzvec(v::SymTrTenArray) =
    v.yz

Base.similar(a::SymTrTenArray) = SymTrTenArray(similar(a.xx),similar(a.xy),similar(a.xz),similar(a.yy),similar(a.yz))
