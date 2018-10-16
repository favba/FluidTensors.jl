
@inline function prop_decomp(A::SymTen,B::SymTen) #Decompose A regarding B
    α = (A:B)/(B:B)
    ApB = α*B
    return α, ApB
end

@inline function inph_decomp(A::SymTen,B::SymTen)
    λ,e = eigvec(B)

    m1 = e[1] ⋅ A ⋅ e[1]
    m2 = e[2] ⋅ A ⋅ e[2]
    m3 = e[3] ⋅ A ⋅ e[3]

    l1 = λ[1]
    l2 = λ[2]
    l3 = λ[3]
    l12 = l1 - l2
    l13 = l1 - l3
    l23 = l2 - l3
    aux = inv(l12*l13*l23)

    α0 = aux*(-l13*l1*l3*m2 + l2*l2*(l3*m1 - l1*m3) + l2*(l1*l1*m3 - l3*l3*m1))
    α1 = aux*(l3*l3*(m1-m2) + l1*l1*(m2-m3) + l2*l2*(m3-m1))
    α2 = aux*(l3*(m2-m1) + l2*(m1-m3) + l1*(m3-m2))

    AinB = m1*symouter(e[1],e[1]) + m2*symouter(e[2],e[2]) + m3*symouter(e[3],e[3])

    return α0, α1, α2, AinB 

end