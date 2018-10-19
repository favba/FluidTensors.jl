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
    S - inv(3)*tr(S)*I

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

@inline Base.adjoint(a::SymTen{T}) where T<:Complex = SymTen{T}(adjoint(a.xx),adjoint(a.xy),adjoint(a.xz),adjoint(a.yy),adjoint(a.yz),adjoint(a.zz))
@inline Base.adjoint(a::SymTen) = a

####### Full Tensors

struct Ten{T<:Number} <: AbstractTen{T}
    xx::T
    yx::T
    zx::T
    xy::T
    yy::T 
    zy::T 
    xz::T
    yz::T 
    zz::T
end


Base.IndexStyle(a::Type{<:Ten}) = Base.IndexLinear()

@inline Base.@propagate_inbounds function Base.getindex(a::Ten,I::Integer) 
    @boundscheck checkbounds(a,I)
    @inbounds getfield(a,I)
end

@inline Base.size(a::Ten) = 
    (3,3)

@inline Base.length(a::Ten) = 
    9

@inline Base.zero(a::Type{Ten{T}}) where {T} = 
    Ten{T}(zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),zero(T))

@inline Ten(xx::T,xy::T,xz::T,yy::T,yz::T,zz::T) where {T} = 
    Ten{T}(xx,xy,xz,yy,yz,zz)

@inline Ten(q,w,e,r,t,y,a,d,f) = 
    Ten(promote(q,w,e,r,t,y,a,d,f)...)

@inline Base.:+(a::Ten{T},b::Ten{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx+b.xx, a.yx+b.yx, a.zx+b.zx, a.xy+b.xy, a.yy+b.yy, a.zy+b.zy, a.xz+b.xz, a.yz + b.yz, a.zz+b.zz)

@inline Base.:-(A::Ten{T}) where T = Ten{T}(-A.xx,-A.yx,-A.zx,-A.xy,-A.yy,-A.zy,-A.xz,-A.yz,-A.zz)

@inline Base.:-(a::Ten{T},b::Ten{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx-b.xx, a.yx-b.yx, a.zx-b.zx, a.xy-b.xy, a.yy - b.yy, a.zy - b.zy, a.xz - b.xz, a.yz - b.yz, a.zz - b.zz)

@inline Base.:*(a::T,b::Ten{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(a*b.xx, a*b.yx, a*b.zx, a*b.xy, a*b.yy, a*b.zy, a*b.xz, a*b.yz, a*b.zz)

@inline Base.:*(b::Ten{T},a::T2) where {T<:Number,T2<:Number} = 
    a*b

@inline Base.:/(b::Ten{T},a::T2) where {T<:Number,T2<:Number} = 
    @fastmath inv(a)*b

@inline Base.:(:)(a::Ten,b::Ten) = 
    @fastmath muladd(a.xx, b.xx, muladd(a.yx, b.yx, muladd(a.zx, b.zx, muladd(a.xy, b.xy, muladd(a.yy, b.yy, muladd(a.zy, b.zy, muladd(a.xz, b.xz, muladd(a.yz, b.yz, a.zz*b.zz))))))))

@inline Base.:(+)(L::LinearAlgebra.UniformScaling,A::Ten) =
    Ten(A.xx + L.λ, A.yx, A.zx, A.xy, A.yy + L.λ, A.zy, A.xz, A.yz, A.zz + L.λ)

@inline Base.:(+)(A::Ten, L::LinearAlgebra.UniformScaling) = L+A

@inline Base.:(-)(A::Ten, L::LinearAlgebra.UniformScaling) =
    Ten(A.xx - L.λ, A.yx, A.zx, A.xy, A.yy - L.λ, A.zy, A.xz, A.yz, A.zz - L.λ)

@inline Base.:(-)(L::LinearAlgebra.UniformScaling,A::Ten) = -(A-L)


@inline LinearAlgebra.tr(a::Ten) =
    @fastmath a.xx + a.yy + a.zz

@inline LinearAlgebra.norm(a::Ten) = 
    @fastmath sqrt(2(a:a))

@inline LinearAlgebra.dot(a::Ten{T},b::Ten{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(a.xx, b.xx, muladd(a.xy, b.yx, a.xz*b.zx)), muladd(a.yx, b.xx, muladd(a.yy, b.yx, a.yz*b.zx)), muladd(a.zx, b.xx, muladd(a.zy, b.yx, a.zz*b.zx)),
                                      muladd(a.xx, b.xy, muladd(a.xy, b.yy, a.xz*b.zy)), muladd(a.yx, b.xy, muladd(a.yy, b.yy, a.yz*b.zy)), muladd(a.zx, b.xy, muladd(a.zy, b.yy, a.zz*b.zy)),
                                      muladd(a.xx, b.xz, muladd(a.xy, b.yz, a.xz*b.zz)), muladd(a.yx, b.xz, muladd(a.yy, b.yz, a.yz*b.zz)), muladd(a.zx, b.xz, muladd(a.zy, b.yz, a.zz*b.zz)))

@inline LinearAlgebra.dot(a::Ten{T},b::Vec{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Vec{promote_type(T,T2)}(muladd(a.xx, b.x, muladd(a.xy, b.y, a.xz*b.z)), 
                                      muladd(a.yx, b.x, muladd(a.yy, b.y, a.yz*b.z)),
                                      muladd(a.zx, b.x, muladd(a.zy, b.y, a.zz*b.z)))

@inline LinearAlgebra.dot(b::Vec{T},a::Ten{T2}) where {T,T2} = 
    @fastmath Vec{promote_type(T,T2)}(muladd(a.xx, b.x, muladd(a.yx, b.y, a.zx*b.z)), 
                                      muladd(a.xy, b.x, muladd(a.yy, b.y, a.zy*b.z)),
                                      muladd(a.xz, b.x, muladd(a.yz, b.y, a.zz*b.z)))

@inline LinearAlgebra.dot(a::SymTen{T},b::SymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(a.xx, b.xx, muladd(a.xy, b.xy, a.xz*b.xz)), muladd(a.xy, b.xx, muladd(a.yy, b.xy, a.yz*b.xz)), muladd(a.xz, b.xx, muladd(a.yz, b.xy, a.zz*b.xz)),
    muladd(a.xx, b.xy, muladd(a.xy, b.yy, a.xz*b.yz)), muladd(a.xy, b.xy, muladd(a.yy, b.yy, a.yz*b.yz)), muladd(a.xz, b.xy, muladd(a.yz, b.yy, a.zz*b.yz)),
    muladd(a.xx, b.xz, muladd(a.xy, b.yz, a.xz*b.zz)), muladd(a.xy, b.xz, muladd(a.yy, b.yz, a.yz*b.zz)), muladd(a.xz, b.xz, muladd(a.yz, b.yz, a.zz*b.zz)))

@inline Base.adjoint(a::Ten{T}) where T = Ten{T}(adjoint(a.xx),adjoint(a.xy),adjoint(a.xz),adjoint(a.yx),adjoint(a.yy),adjoint(a.yz),adjoint(a.zx),adjoint(a.zy),adjoint(a.zz))

@inline outer(a::Vec,b::Vec) = #symmetric part of the outer product of two vectors
    @fastmath Ten(a.x*b.x, a.y*b.x, a.z*b.x,
                  a.x*b.y, a.y*b.y, a.z*b.y,
                  a.x*b.z, a.y*b.z, a.z*b.z)

const ⊗ = outer

@inline traceless(S::Ten) =
    S - inv(3)*LinearAlgebra.tr(S)*LinearAlgebra.I

function Base.:^(t::Ten{T},n::Integer) where {T}
    n == 1 && return t
    n == 2 && return t⋅t
    return t⋅(t^(n-1))
end

@inline function Base.literal_pow(::typeof(Base.:^),t::Ten{T},::Val{2}) where {T<:AbstractFloat}
    return t⋅t
end

@inline SymTen(t::Ten) = SymTen(t.xx, (t.xy + t.yx)/2, (t.xz + t.zx)/2, t.yy, (t.zy + t.yz)/2, t.zz)
@inline Ten(a::SymTen) = Ten(a.xx,a.xy,a.xz,a.xy,a.yy,a.yz,a.xz,a.yz,a.zz)

####### AntiSym Tensors

struct AntiSymTen{T<:Number} <: AbstractTen{T}
    xy::T
    xz::T
    yz::T 
end


Base.IndexStyle(a::Type{<:AntiSymTen}) = Base.IndexLinear()

@inline Base.@propagate_inbounds function Base.getindex(a::AntiSymTen{T},I::Integer) where T
    @boundscheck checkbounds(a,I)
    (I == 1 || I == 5 || I == 9) && return zero(T)
    I == 2 && return -a.xy
    I == 3 && return -a.xz
    I == 4 && return a.xy
    I == 6 && return -a.yz
    I == 7 && return a.xz
    I == 8 && return a.yz
end

@inline Base.size(a::AntiSymTen) = 
    (3,3)

@inline Base.length(a::AntiSymTen) = 
    9

@inline Base.zero(a::Type{AntiSymTen{T}}) where {T} = 
    AntiSymTen{T}(zero(T),zero(T),zero(T))

@inline AntiSymTen(xx::T,xy::T,xz::T) where {T} = 
    AntiSymTen{T}(xx,xy,xz)

@inline AntiSymTen(q,w,e) = 
    AntiSymTen(promote(q,w,e)...)

@inline Base.:+(a::AntiSymTen{T},b::AntiSymTen{T2}) where {T,T2} = 
    @fastmath AntiSymTen{promote_type(T,T2)}(a.xy+b.xy, a.xz+b.xz, a.yz + b.yz)

@inline Base.:-(A::AntiSymTen{T}) where T = AntiSymTen{T}(-A.xy,-A.xz,-A.yz)

@inline Base.:-(a::AntiSymTen{T},b::AntiSymTen{T2}) where {T,T2} = 
    @fastmath AntiSymTen{promote_type(T,T2)}(a.xy-b.xy, a.xz - b.xz, a.yz - b.yz)

@inline Base.:*(a::T,b::AntiSymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath AntiSymTen{promote_type(T,T2)}(a*b.xy, a*b.xz, a*b.yz)

@inline Base.:*(b::AntiSymTen{T},a::T2) where {T<:Number,T2<:Number} = 
    a*b

@inline Base.:/(b::AntiSymTen{T},a::T2) where {T<:Number,T2<:Number} = 
    @fastmath inv(a)*b

@inline Base.:(:)(a::AntiSymTen,b::AntiSymTen) = 
    @fastmath 2muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*a.yz))

@inline Base.:(+)(L::LinearAlgebra.UniformScaling,A::AntiSymTen) =
    Ten(L.λ, -A.xy, -A.xz, A.xy, L.λ, -A.yz, A.xz, A.yz, L.λ)

@inline Base.:(+)(A::AntiSymTen, L::LinearAlgebra.UniformScaling) = L+A

@inline Base.:(-)(A::AntiSymTen, L::LinearAlgebra.UniformScaling) =
    Ten(-L.λ, -A.xy, -A.xz, A.xy, -L.λ, -A.yz, A.xz, A.yz, -L.λ)

@inline Base.:(-)(L::LinearAlgebra.UniformScaling,A::AntiSymTen) = -(A-L)


@inline LinearAlgebra.tr(a::AntiSymTen{T}) where T = zero(T)

@inline LinearAlgebra.norm(a::AntiSymTen) = 
    @fastmath sqrt(2(a:a))

@inline LinearAlgebra.dot(a::AntiSymTen{T},b::AntiSymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(-muladd(a.xy, b.xy, a.xz*b.xz), -a.yz*b.xz, a.yz*b.xy,
                                      -a.xz*b.yz, -muladd(a.xy,b.xy,a.yz*b.yz), -a.xz*b.xy,
                                      a.xy*b.yz, -a.xy*b.xz, -muladd(a.xz,b.xz,a.yz*b.yz))

@inline LinearAlgebra.dot(a::AntiSymTen{T},b::Vec{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Vec{promote_type(T,T2)}(muladd(a.xy, b.y, a.xz*b.z), 
                                      muladd(a.yz, b.z, -a.xy*b.x),
                                      -muladd(a.xz, b.x, a.yz*b.y))

@inline LinearAlgebra.dot(b::Vec,a::AntiSymTen) = -a⋅b

@inline Base.adjoint(a::AntiSymTen{T}) where T = AntiSymTen{T}(adjoint(-a.xy),adjoint(-a.xz),adjoint(-a.yz))

@inline traceless(S::AntiSymTen) = S

Base.:^(t::AntiSymTen{T},n::Integer) where {T} = Ten(t)^n

@inline function Base.literal_pow(::typeof(Base.:^),a::AntiSymTen{T},::Val{2}) where {T<:AbstractFloat}
    return @fastmath SymTen{T}(-muladd(a.xy,a.xy,a.xz^2), -a.xz*a.yz, a.xz*a.yz, -muladd(a.xy,a.xy,a.yz^2), -a.xy*a.xz, -muladd(a.xz,a.xz,a.yz^2))
end

@inline SymTen(a::AntiSymTen{T}) where {T} = zero(SymTen{T})
@inline AntiSymTen(a::SymTen{T}) where {T} = zero(AntiSymTen{T})
@inline Ten(a::AntiSymTen{T}) where {T} = Ten{T}(zero(T),-a.xy,-a.xz,a.xy,zero(T),-a.yz,a.xz,a.yz,zero(T))
@inline AntiSymTen(t::Ten) = AntiSymTen((t.xy - t.yx)/2, (t.xz - t.zx)/2, (t.yz - t.zy)/2)
@inline Vec(a::AntiSymTen) = Vec(-a.yz,a.xz,-a.xy)
@inline AntiSymTen(a::Vec) = AntiSymTen(-a.z,a.y,-a.x)

@inline Lie(a::SymTen,b::AntiSymTen) = Lie(a,-Vec(b))


############################## Interoperation

@inline Base.:+(a::AntiSymTen{T},b::SymTen{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(b.xx, b.xy-a.xy, b.xz-a.xz, a.xy+b.xy, b.yy, b.yz - a.yz, a.xz+b.xz, a.yz+b.yz, b.zz)

@inline Base.:+(a::AntiSymTen{T},b::Ten{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(b.xx, b.yx-a.xy, b.zx-a.xz, a.xy+b.xy, b.yy, b.zy - a.yz, a.xz+b.xz, a.yz+b.yz, b.zz)

@inline Base.:+(a::SymTen{T},b::Ten{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx+b.xx, a.xy+b.yx, a.xz+b.zx, a.xy+b.xy, a.yy+b.yy, a.yz+b.zy, a.xz+b.xz, a.yz + b.yz, a.zz+b.zz)

@inline Base.:+(b::SymTen,a::AntiSymTen) = a+b

@inline Base.:+(b::Ten,a::AntiSymTen) = a+b

@inline Base.:+(b::Ten,a::SymTen) = a+b

@inline Base.:-(a::AntiSymTen{T},b::SymTen{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(-b.xx, -a.xy-b.xy, -a.xz-b.xz, a.xy-b.xy, -b.yy, -a.yz - b.yz, a.xz - b.xz, a.yz - b.yz, -b.zz)

@inline Base.:-(a::SymTen{T},b::AntiSymTen{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx, a.xy+b.xy, a.xz+b.xz, a.xy-b.xy, a.yy, a.yz + b.yz, a.xz - b.xz, a.yz - b.yz, a.zz)

@inline Base.:-(a::Ten{T},b::AntiSymTen{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx, a.yx+b.xy, a.zx+b.xz, a.xy-b.xy, a.yy, a.zy + b.yz, a.xz - b.xz, a.yz - b.yz, a.zz)

@inline Base.:-(a::AntiSymTen{T},b::Ten{T2}) where {T,T2} = -(b-a) 

@inline Base.:-(a::Ten{T},b::SymTen{T2}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(a.xx-b.xx, a.yx-b.xy, a.zx-b.xz, a.xy-b.xy, a.yy - b.yy, a.zy - b.yz, a.xz - b.xz, a.yz - b.yz, a.zz - b.zz)

@inline Base.:-(a::SymTen{T},b::Ten{T2}) where {T,T2} = -(b-a) 

@inline LinearAlgebra.dot(a::Ten{T},b::SymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(a.xx, b.xx, muladd(a.xy, b.xy, a.xz*b.xz)), muladd(a.yx, b.xx, muladd(a.yy, b.xy, a.yz*b.xz)), muladd(a.zx, b.xx, muladd(a.zy, b.xy, a.zz*b.xz)),
                                      muladd(a.xx, b.xy, muladd(a.xy, b.yy, a.xz*b.yz)), muladd(a.yx, b.xy, muladd(a.yy, b.yy, a.yz*b.yz)), muladd(a.zx, b.xy, muladd(a.zy, b.yy, a.zz*b.yz)),
                                      muladd(a.xx, b.xz, muladd(a.xy, b.yz, a.xz*b.zz)), muladd(a.yx, b.xz, muladd(a.yy, b.yz, a.yz*b.zz)), muladd(a.zx, b.xz, muladd(a.zy, b.yz, a.zz*b.zz)))

@inline LinearAlgebra.dot(a::SymTen{T},b::Ten{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(a.xx, b.xx, muladd(a.xy, b.yx, a.xz*b.zx)), muladd(a.xy, b.xx, muladd(a.yy, b.yx, a.yz*b.zx)), muladd(a.xz, b.xx, muladd(a.yz, b.yx, a.zz*b.zx)),
    muladd(a.xx, b.xy, muladd(a.xy, b.yy, a.xz*b.zy)), muladd(a.xy, b.xy, muladd(a.yy, b.yy, a.yz*b.zy)), muladd(a.xz, b.xy, muladd(a.yz, b.yy, a.zz*b.zy)),
    muladd(a.xx, b.xz, muladd(a.xy, b.yz, a.xz*b.zz)), muladd(a.xy, b.xz, muladd(a.yy, b.yz, a.yz*b.zz)), muladd(a.xz, b.xz, muladd(a.yz, b.yz, a.zz*b.zz)))

@inline LinearAlgebra.dot(a::Ten{T},b::AntiSymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(-muladd(a.xy,b.xy,a.xz*b.xz), -muladd(a.yy,b.xy, a.yz*b.xz), -muladd(a.zy,b.xy, a.zz*b.xz),
                                      a.xx*b.xy - a.xz*b.yz, a.yx*b.xy - a.yz*b.yz, a.zx*b.xy - a.zz*b.yz,
                                      muladd(a.xx,b.xz,a.xy*b.yz), muladd(a.yx,b.xz,a.yy*b.yz), muladd(a.zx,b.xz, a.zy*b.yz))


@inline LinearAlgebra.dot(b::AntiSymTen{T2},a::Ten{T}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(b.xy, a.yx, b.xz*a.zx), -a.xx*b.xy+a.zx*b.yz, -muladd(a.xx,b.xz,a.yx*b.yz),
                                      muladd(a.yy,b.xy,a.zy*b.xz), -a.xy*b.xy+a.zy*b.yz, -muladd(a.xy,b.xz,a.yy*b.yz),
                                      muladd(a.yz,b.xy,a.zz*b.xz), -a.xz*b.xy+a.zz*b.yz, -muladd(a.xz,b.xz,a.yz*b.yz))

@inline LinearAlgebra.dot(a::SymTen{T},b::AntiSymTen{T2}) where {T<:Number,T2<:Number} = 
    @fastmath Ten{promote_type(T,T2)}(-muladd(a.xy,b.xy,a.xz*b.xz), -muladd(a.yy,b.xy, a.yz*b.xz), -muladd(a.yz,b.xy, a.zz*b.xz),
                                      a.xx*b.xy - a.xz*b.yz, a.xy*b.xy - a.yz*b.yz, a.xz*b.xy - a.zz*b.yz,
                                      muladd(a.xx,b.xz,a.xy*b.yz), muladd(a.xy,b.xz,a.yy*b.yz), muladd(a.xz,b.xz, a.yz*b.yz))


@inline LinearAlgebra.dot(b::AntiSymTen{T2},a::SymTen{T}) where {T,T2} = 
    @fastmath Ten{promote_type(T,T2)}(muladd(a.xy,b.xy,a.xz*b.xz), -a.xx*b.xy+a.xz*b.yz, -muladd(a.xx,b.xz,a.xy*b.yz),
                                      muladd(a.yy,b.xy,a.yz*b.xz), -a.xy*b.xy+a.yz*b.yz, -muladd(a.xy,b.xz,a.yy*b.yz),
                                      muladd(a.yz,b.xy,a.zz*b.xz), -a.xz*b.xy+a.zz*b.yz, -muladd(a.xz,b.xz,a.yz*b.yz))


include("tensorarrays.jl")
