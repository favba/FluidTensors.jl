function eig(t::SymTen{T}) where T<:AbstractFloat

    e11 = t.xx
    e12 = t.xy
    e13 = t.xz
    e22 = t.yy
    e23 = t.yz
    e33 = t.zz

    #=@fastmath=# begin
        p1 = muladd(e12, e12, muladd(e13, e13, e23*e23))
        q = (e11 + e22 + e33)/3
        p2 = (e11-q)^2 + (e22-q)^2 + (e33-q)^2 + 2*p1
        p = @fastmath sqrt(p2/6)
        r = ((e11-q)*(e22-q)*(e33-q) - (e11-q)*(e23^2) - (e12^2)*(e33-q) + 2*(e12*e13*e23) - (e13^2)*(e22-q))/(2*p*p*p)
  
        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.

        ϕ =  ifelse(r <= -1, T(π/3), ifelse(r >= 1, zero(T), acos(r)/3))
  
          # the eigenvalues satisfy eig[3] <= eig[2] <= eig[1]
        eig1 = q + 2*p*cos(ϕ)
        eig3 = q + 2*p*cos(ϕ+(2*π/3))
        eig2 = 3*q - eig1 - eig3     # since trace(E) = eig[1] + eig[2] + eig[3] = 3q
    end

    return (eig1,eig2,eig3)
end

function eigvec(t::SymTen{T},eig::NTuple{3,T}) where {T<:AbstractFloat}
    e11 = t.xx
    e12 = t.xy
    e13 = t.xz
    e22 = t.yy
    e23 = t.yz
    e33 = t.zz

    bla = ((e22 - eig[1])*(e33 - eig[1]) - e23*e23)
    if bla != 0
        eigv11 = oneunit(T)
        eigv12 = (e23*e13 - (e33-eig[1])*e12)/bla
        eigv13 = (-e13 -e23*eigv12)/(e33-eig[1])
        aux = sqrt(1 + eigv12^2 + eigv13^2)
        eigv11 = 1/aux
        eigv12 = eigv12/aux
        eigv13 = eigv13/aux
    else
        bla = ((e11 - eig[1])*(e22 - eig[1]) - e12*e12)
        eigv13 = oneunit(T)
        eigv11 = (e23*e12 - (e22-eig[1])*e13)/bla
        eigv12 = (-e23 -e12*eigv11)/(e22-eig[1])
        aux = sqrt(1 + eigv12^2 + eigv11^2)
        eigv11 = eigv11/aux
        eigv12 = eigv12/aux
        eigv13 = 1/aux
    end
    bla = ((e22 - eig[2])*(e33 - eig[2]) - e23*e23)
    if bla != 0
        eigv21 = oneunit(T)
        eigv22 = (e23*e13 - (e33-eig[2])*e12)/bla
        eigv23 = (-e13 -e23*eigv22)/(e33-eig[2])
        aux = sqrt(1 + eigv22^2 + eigv23^2)
        eigv21 = 1/aux
        eigv22 = eigv22/aux
        eigv23 = eigv23/aux
    else
        bla = ((e11 - eig[2])*(e22 - eig[2]) - e12*e12)
        eigv23 = oneunit(T)
        eigv21 = (e23*e12 - (e22-eig[2])*e13)/bla
        eigv22 = (-e23 -e12*eigv21)/(e22-eig[2])
        aux = sqrt(1 + eigv22^2 + eigv21^2)
        eigv21 = eigv21/aux
        eigv22 = eigv22/aux
        eigv23 = 1.0/aux
    end

    eigv1 = Vec(eigv11,eigv12,eigv13)
    eigv2 = Vec(eigv21,eigv22,eigv23)
    eigv3 = LinearAlgebra.cross(eigv1,eigv2)

    return eigv1,eigv2,eigv3
end

function eigvec(t::SymTen{T}) where {T<:AbstractFloat} 
    eigs = eig(t)
    return eigs, eigvec(t,eigs)
end