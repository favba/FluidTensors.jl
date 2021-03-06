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

struct SymTenArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N},C<:AbstractArray{T,N},D<:AbstractArray{T,N},E<:AbstractArray{T,N},F<:AbstractArray{T,N}} <: AbstractSymTenArray{T,N}
    xx::A
    xy::B
    xz::C
    yy::D
    yz::E
    zz::F
end

@generated function Base.IndexStyle(::Type{<:SymTenArray{T,N,A,B,C,D,E,F}}) where {T,N,A,B,C,D,E,F} 
    indexstyle = all(x->(IndexStyle(x)===IndexLinear()),(A,B,C,D,E,F)) ? IndexLinear() : IndexCartesian()
    return :($indexstyle)
end

@inline SymTenArray{T}(dims::Vararg{Int,N}) where {T,N} = SymTenArray(zeros(T,dims...),zeros(T,dims...),zeros(T,dims...),zeros(T,dims...),zeros(T,dims...),zeros(T,dims...))
@inline SymTenArray(dims::Vararg{Int,N}) where {N} = SymTenArray{Float64}(dims...)

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

Base.read!(io::NTuple{6,A},a::SymTenArray) where{A<:Union{<:IO,<:AbstractString}} = (read!(io[1],a.xx); read!(io[2],a.xy); read!(io[3],a.xz); read!(io[4],a.yy); read!(io[5],a.yz); read!(io[6],a.zz))
Base.write(io::NTuple{6,A},a::SymTenArray) where{A<:Union{<:IO,<:AbstractString}} = (write(io[1],a.xx); write(io[2],a.xy); write(io[3],a.xz); write(io[4],a.yy); write(io[5],a.yz); write(io[6],a.zz))

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

@generated function Base.IndexStyle(::Type{<:SymTrTenArray{T,N,A,B,C,D,E}}) where {T,N,A,B,C,D,E} 
    indexstyle = all(x->(IndexStyle(x)===IndexLinear()),(A,B,C,D,E)) ? IndexLinear() : IndexCartesian()
    return :($indexstyle)
end

@inline SymTrTenArray{T}(dims::Vararg{Int,N}) where {T,N} = SymTrTenArray(zeros(T,dims...),zeros(T,dims...),zeros(T,dims...),zeros(T,dims...),zeros(T,dims...))
@inline SymTrTenArray(dims::Vararg{Int,N}) where {N} = SymTrTenArray{Float64}(dims...)
    
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

Base.read!(io::NTuple{5,A},a::SymTrTenArray) where{A<:Union{<:IO,<:AbstractString}} = (read!(io[1],a.xx); read!(io[2],a.xy); read!(io[3],a.xz); read!(io[4],a.yy); read!(io[5],a.yz))
Base.write(io::NTuple{5,A},a::SymTrTenArray) where{A<:Union{<:IO,<:AbstractString}} = (write(io[1],a.xx); write(io[2],a.xy); write(io[3],a.xz); write(io[4],a.yy); write(io[5],a.yz))


abstract type AbstractAntiSymTenArray{T,N} <: AbstractArray{AntiSymTen{T},N} end

@inline Base.@propagate_inbounds function Base.getindex(v::AbstractAntiSymTenArray{T,N},i::Int) where {T,N}
    xy = xyvec(v)
    xz = xzvec(v)
    yz = yzvec(v)
    xyv = xy[i]
    xzv = xz[i]
    yzv = yz[i]
    return AntiSymTen{T}(xyv,xzv,yzv)
end
    
@inline Base.@propagate_inbounds function Base.getindex(v::AbstractAntiSymTenArray{T,N},I::Vararg{Int,N}) where {T,N}
    xy = xyvec(v)
    xz = xzvec(v)
    yz = yzvec(v)
    xyv = xy[I...]
    xzv = xz[I...]
    yzv = yz[I...]
    return AntiSymTen{T}(xyv,xzv,yzv)
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractAntiSymTenArray,ten,i::Int)
    xy = xyvec(v)
    xz = xzvec(v)
    yz = yzvec(v)
    setindex!(xy,ten.xy,i)
    setindex!(xz,ten.xz,i)
    setindex!(yz,ten.yz,i)
    return v
end
    
@inline Base.@propagate_inbounds function Base.setindex!(v::AbstractAntiSymTenArray{T,N},vec,I::Vararg{Int,N}) where {T,N}
    xy = xyvec(v)
    xz = xzvec(v)
    yz = yzvec(v)
    setindex!(xy,ten.xy,I...)
    setindex!(xz,ten.xz,I...)
    setindex!(yz,ten.yz,I...)
    return v
end

Base.size(v::AbstractAntiSymTenArray) =
    size(xyvec(v))

struct AntiSymTenArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N},C<:AbstractArray{T,N}} <: AbstractAntiSymTenArray{T,N}
    xy::A
    xz::B
    yz::C
end

@generated function Base.IndexStyle(::Type{<:AntiSymTenArray{T,N,A,B,C}}) where {T,N,A,B,C} 
    indexstyle = all(x->(IndexStyle(x)===IndexLinear()),(A,B,C)) ? IndexLinear() : IndexCartesian()
    return :($indexstyle)
end

@inline AntiSymTenArray{T}(dims::Vararg{Int,N}) where {T,N} = AntiSymTenArray(zeros(T,dims...),zeros(T,dims...),zeros(T,dims...))
@inline AntiSymTenArray(dims::Vararg{Int,N}) where {N} = AntiSymTenArray{Float64}(dims...)
    
@inline xyvec(v::AntiSymTenArray) =
    v.xy
@inline xzvec(v::AntiSymTenArray) =
    v.xz
@inline yzvec(v::AntiSymTenArray) =
    v.yz

Base.similar(a::AntiSymTenArray) = AntiSymTenArray(similar(a.xy),similar(a.xz),similar(a.yz))

Base.read!(io::NTuple{3,A},a::AntiSymTenArray) where{A<:Union{<:IO,<:AbstractString}} = (read!(io[1],a.xy); read!(io[2],a.xz); read!(io[3],a.yz))
Base.write(io::NTuple{3,A},a::AntiSymTenArray) where{A<:Union{<:IO,<:AbstractString}} = (write(io[1],a.xy); write(io[2],a.xz); write(io[3],a.yz))