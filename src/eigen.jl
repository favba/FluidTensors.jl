@inline function eig(t::SymTen{T}) where T<:AbstractFloat

    e11 = t.xx
    e12 = t.xy
    e13 = t.xz
    e22 = t.yy
    e23 = t.yz
    e33 = t.zz

    p1 = muladd(e12, e12, muladd(e13, e13, e23*e23))
    q = (e11 + e22 + e33)/3
    p2 = (e11-q)^2 + (e22-q)^2 + (e33-q)^2 + 2*p1
    p = @fastmath sqrt(p2/6)
    r = ((e11-q)*(e22-q)*(e33-q) - (e11-q)*(e23^2) - (e12^2)*(e33-q) + 2*(e12*e13*e23) - (e13^2)*(e22-q))/(2*p*p*p)
  
    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.

    if r <= -1
        ϕ =  T(π/3)
    elseif r >= 1
        ϕ = zero(T)
    else
        ϕ = acos(r)/3
    end
  
    # the eigenvalues satisfy eig.z >= eig.y >= eig.x
    eig3 = q + 2*p*cos(ϕ)
    # cos(x+y) = cos(x)*cos(y) - sin(x)*sin(y)
    eig1 = q + 2*p*cos(ϕ+(2*π/3))  # q - 2*p*(cos(ϕ)/2 + (√3/2)sin(ϕ))
    eig2 = 3*q - eig1 - eig3     # since trace(E) = eig.x + eig.y + eig.z = 3q

    return (eig1,eig2,eig3)
end

@inline function eig(t::SymTen)

    e11 = t.xx
    e12 = t.xy
    e13 = t.xz
    e22 = t.yy
    e23 = t.yz
    e33 = t.zz

    p1 = muladd(e12, e12, muladd(e13, e13, e23*e23))
    q = (e11 + e22 + e33)/3
    p2 = (e11-q)^2 + (e22-q)^2 + (e33-q)^2 + 2*p1
    p = sqrt(p2/6)
    r = ((e11-q)*(e22-q)*(e33-q) - (e11-q)*(e23^2) - (e12^2)*(e33-q) + 2*(e12*e13*e23) - (e13^2)*(e22-q))/(2*p*p*p)
  
    ϕ = acos(r)/3
  
    # the eigenvalues satisfy eig.z >= eig.y >= eig.x
    eig3 = q + 2*p*cos(ϕ)
    eig1 = q + 2*p*cos(ϕ+(2*π/3))
    eig2 = 3*q - eig1 - eig3     # since trace(E) = eig.x + eig.y + eig.z = 3q

    return (eig1,eig2,eig3)
end

function eigvec(t::SymTen{T}) where {T<:AbstractFloat}
    S11 = t.xx
    S12 = t.xy
    S13 = t.xz
    S22 = t.yy
    S23 = t.yz
    S33 = t.zz

    p1 = muladd(S12, S12, muladd(S13, S13, S23*S23))

    if (p1 == 0) # diagonal tensor
        v1 = Vec{T}(1,0,0)
        v2 = Vec{T}(0,1,0)
        v3 = Vec{T}(0,0,1)
        if S11 < S22
            if S22 < S33
                return (S11, S22, S33), (v1, v2, v3)
            elseif S33 < S11
                return (S33, S11, S22), (v3, v1, v2)
            else
                return (S11, S33, S22), (v1, v3, v2)
            end
        else #S22 < S11
            if S11 < S33
                return (S22, S11, S33), (v2, v1, v3)
            elseif S33 < S22
                return (S33, S22, S11), (v3, v2, v1)
            else
                return (S22, S33, S11), (v2, v3, v1)
            end
        end
    end

    q = (S11 + S22 + S33)/3
    p2 = (S11-q)^2 + (S22-q)^2 + (S33-q)^2 + 2*p1
    p = @fastmath sqrt(p2/6)
    r = ((S11-q)*(S22-q)*(S33-q) - (S11-q)*(S23^2) - (S12^2)*(S33-q) + 2*(S12*S13*S23) - (S13^2)*(S22-q))/(2*p*p*p)
  
    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.

    if r <= -1
        ϕ =  T(π/3)
    elseif r >= 1
        ϕ = zero(T)
    else
        ϕ = acos(r)/3
    end  

    # the eigenvalues satisfy eig.z >= eig.y >= eig.x
    λ3 = q + 2*p*cos(ϕ)
    λ1 = q + 2*p*cos(ϕ+(2*π/3))
    λ2 = 3*q - λ1 - λ3     # since trace(E) = eig.x + eig.y + eig.z = 3q


    ######################### This part was copied from https://github.com/KristofferC/Tensors.jl/blob/master/src/eigen.jl #################################

    if r > 0
        (λ1, λ3) = (λ3, λ1)
    end
      # Calculate the first eigenvector
        # This should be orthogonal to these three rows of A - λ1*I
        # Use all combinations of cross products and choose the "best" one
    r₁ = Vec(S11 - λ1, S12, S13)
    r₂ = Vec(S12, S22 - λ1, S23)
    r₃ = Vec(S13, S23, S33 - λ1)
    n₁ = r₁ ⋅ r₁
    n₂ = r₂ ⋅ r₂
    n₃ = r₃ ⋅ r₃

    r₁₂ = r₁ × r₂
    r₂₃ = r₂ × r₃
    r₃₁ = r₃ × r₁
    n₁₂ = r₁₂ ⋅ r₁₂
    n₂₃ = r₂₃ ⋅ r₂₃
    n₃₁ = r₃₁ ⋅ r₃₁

    # we want best angle so we put all norms on same footing
    # (cheaper to multiply by third nᵢ rather than divide by the two involved)
    if n₁₂ * n₃ > n₂₃ * n₁
        if n₁₂ * n₃ > n₃₁ * n₂
            @fastmath ϕ1 = r₁₂ / sqrt(n₁₂)
        else
            @fastmath ϕ1 = r₃₁ / sqrt(n₃₁)
        end
    else
        if n₂₃ * n₁ > n₃₁ * n₂
            @fastmath ϕ1 = r₂₃ / sqrt(n₂₃)
        else
            @fastmath ϕ1 = r₃₁ / sqrt(n₃₁)
        end
    end

    # Calculate the second eigenvector
    # This should be orthogonal to the previous eigenvector and the three
    # rows of A - λ2*I. However, we need to "solve" the remaining 2x2 subspace
    # problem in case the cross products are identically or nearly zero

    # The remaing 2x2 subspace is:
    if abs(ϕ1.x) < abs(ϕ1.y) # safe to set one component to zero, depending on this
        @fastmath orthogonal1 = Vec(-ϕ1.z, zero(T), ϕ1.x) / sqrt(abs2(ϕ1.x) + abs2(ϕ1.z))
    else
        @fastmath orthogonal1 = Vec(zero(T), ϕ1.z, -ϕ1.y) / sqrt(abs2(ϕ1.y) + abs2(ϕ1.z))
    end
    orthogonal2 = ϕ1 × orthogonal1

    # The projected 2x2 eigenvalue problem is C x = 0 where C is the projection
    # of (A - λ2*I) onto the subspace {orthogonal1, orthogonal2}
    a_orth1_1 = S11 * orthogonal1.x + S12 * orthogonal1.y + S13 * orthogonal1.z
    a_orth1_2 = S12 * orthogonal1.x + S22 * orthogonal1.y + S23 * orthogonal1.z
    a_orth1_3 = S13 * orthogonal1.x + S23 * orthogonal1.y + S33 * orthogonal1.z

    a_orth2_1 = S11 * orthogonal2.x + S12 * orthogonal2.y + S13 * orthogonal2.z
    a_orth2_2 = S12 * orthogonal2.x + S22 * orthogonal2.y + S23 * orthogonal2.z
    a_orth2_3 = S13 * orthogonal2.x + S23 * orthogonal2.y + S33 * orthogonal2.z

    c11 = orthogonal1.x*a_orth1_1 + orthogonal1.y*a_orth1_2 + orthogonal1.z*a_orth1_3 - λ2
    c12 = orthogonal1.x*a_orth2_1 + orthogonal1.y*a_orth2_2 + orthogonal1.z*a_orth2_3
    c22 = orthogonal2.x*a_orth2_1 + orthogonal2.y*a_orth2_2 + orthogonal2.z*a_orth2_3 - λ2

    # Solve this robustly (some values might be small or zero)
    c11² = abs2(c11)
    c12² = abs2(c12)
    c22² = abs2(c22)
    if c11² >= c22²
        if c11² > 0 || c12² > 0
            if c11² >= c12²
                tmp = c12 / c11
                @fastmath p2 = inv(sqrt(1 + abs2(tmp)))
                p1 = tmp * p2
            else
                tmp = c11 / c12 # TODO check for compex input
                @fastmath p1 = inv(sqrt(1 + abs2(tmp)))
                p2 = tmp * p1
            end
            ϕ2 = p1*orthogonal1 - p2*orthogonal2
        else # c11 == 0 && c12 == 0 && c22 == 0 (smaller than c11)
            ϕ2 = orthogonal1
        end
    else
        if c22² >= c12²
            tmp = c12 / c22
            @fastmath p1 = inv(sqrt(1 + abs2(tmp)))
            p2 = tmp * p1
        else
            tmp = c22 / c12
            @fastmath p2 = inv(sqrt(1 + abs2(tmp)))
            p1 = tmp * p2
        end
        ϕ2 = p1*orthogonal1 - p2*orthogonal2
    end


    # The third eigenvector is a simple cross product of the other two
    ϕ3 = ϕ1 × ϕ2 # should be normalized already

    ###############################################################################################################################333
    if r > 0
        (λ1, λ3) = (λ3, λ1)
        (ϕ1, ϕ3) = (-ϕ3, ϕ1) # - sign so e3 = cross(e1,e2) is always true
    end

    return (λ1,λ2,λ3),(ϕ1,ϕ2,ϕ3)
end

@inline stress_state(a::Number,b::Number,c::Number) = (-3*sqrt(6)*a*b*c)/((a^2+b^2+c^2)^1.5)
@inline stress_state(t::SymTen) = stress_state(eig(t)...)