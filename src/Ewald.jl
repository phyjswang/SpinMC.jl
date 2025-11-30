using LinearAlgebra
using StaticArrays
using SpecialFunctions
using TimerOutputs

# based on https://github.com/SunnySuite/Sunny.jl/blob/main/src/System/Ewald.jl

# Tensor product of 3-vectors
(⊗)(a::SVector{3, Float64},b::SVector{3, Float64}) = reshape(kron(a,b), 3, 3)

# a purely geometric quantity, depending only on the system size
@timeit_debug function addDipolarInteractions!(lattice::Lattice{D,N}) where {D,N}
    println("Constructing dipolar interactions tensor, this may take some time!")

    na = lattice.length

    # Superlattice vectors and reciprocals for the full system volume
    if D == 1
        v1 = lattice.unitcell.primitive
        v1 = SVector{3}(v1...,0,0)
        v2 = SVector{3}(0,1,0)
        v3 = SVector{3}(0,0,1)
        lsprimitive = (v1, v2, v3)
        sys_size = diagm(SVector{3}(lattice.size...,1,1))
    elseif D == 2
        v1, v2 = lattice.unitcell.primitive
        v1 = SVector{3}(v1...,0)
        v2 = SVector{3}(v2...,0)
        v3 = SVector{3}(0,0,1)
        lsprimitive = (v1, v2, v3)
        sys_size = diagm(SVector{3}(lattice.size...,1))
    else
        lsprimitive = SVector{3}.(lattice.unitcell.primitive)
        sys_size = diagm(SVector{3}(lattice.size))
    end
    matprimitive = hcat(lsprimitive...)
    latvecs = matprimitive * sys_size
    recipvecs = inv(latvecs') .* 2π

    # Precalculate constants
    I₃ = SMatrix{3, 3, Float64, 9}(I)
    V = det(latvecs)
    L = cbrt(V)
    # Roughly balances the real and Fourier space costs. Note that σ = 1/(√2 λ)
    σ = L/D
    σ² = σ*σ
    σ³ = σ^3
    # Corresponding to c0=6 in Ewalder.jl. Should give ~13 digits of accuracy.
    rmax = 6√2 * σ
    kmax = 6√2 / σ

    nmax = map(eachcol(latvecs), eachcol(recipvecs)) do a, b
        round(Int, rmax / (a⋅normalize(b)) + 1e-6) + 1
    end
    lsn = [-nmax[i]:nmax[i] for i in 1:D]
    mmax = map(eachcol(latvecs), eachcol(recipvecs)) do a, b
        round(Int, kmax / (b⋅normalize(a)) + 1e-6)
    end
    lsm = [-mmax[i]:mmax[i] for i in 1:D]

    # nmax and mmax should be balanced here
    # println("nmax $nmax mmax $mmax")

    for j in 1:na, i in 1:na
        acc = zeros(3,3)
        Δr = SVector{3}((lattice.sitePositions[j] .- lattice.sitePositions[i])...,zeros(3-D)...)

        #####################################################
        ## Real space part
        # for n1 = -nmax[1]:nmax[1], n2 = -nmax[2]:nmax[2], n3 = -nmax[3]:nmax[3]
        for ni in Iterators.product(lsn...)
            n = SVector{3, Float64}(ni...,zeros(3-D)...)
            rvec = Δr .+ latvecs * n
            r² = rvec⋅rvec
            if 0 < r² <= rmax*rmax
                r = √r²
                r³ = r²*r
                rhat = rvec/r
                erfc0 = erfc(r/(√2*σ))
                gauss0 = √(2/π) * (r/σ) * exp(-r²/2σ²)
                acc .+= (I₃/r³) * (erfc0 + gauss0) .- (3(rhat⊗rhat)/r³) * (erfc0 + (1+r²/3σ²) * gauss0)
            end
        end

        #####################################################
        ## Fourier space part
        # for m1 = -mmax[1]:mmax[1], m2 = -mmax[2]:mmax[2], m3 = -mmax[3]:mmax[3]
        for mi in Iterators.product(lsm...)
            m = SVector{3, Float64}(mi...,zeros(3-D)...)
            k = recipvecs * m
            k² = k⋅k

            ϵ² = 1e-16
            if k² <= ϵ²
                # Consider including a surface dipole term as in S. W. DeLeeuw,
                # J. W. Perram, and E. R. Smith, Proc. R. Soc. Lond. A 373,
                # 27-56 (1980). For a spherical geometry, this term might be:
                # acc += (μ0/2V) * I₃
            elseif ϵ² < k² <= kmax*kmax
                phase = cos(-k⋅Δr)
                acc .+= 4π * phase * (1/V) * (exp(-σ²*k²/2) / k²) * (k⊗k)
            end
        end

        #####################################################
        ## Remove self energies
        if all(iszero.(Δr))
            acc .+= - I₃ * 2^(1/2) / (3(π)^(1/2)*σ³)
        end

        # multiply the dipolar interaction strength
        # acc .*= lattice.unitcell.dipolar

        lattice.interactionDipolar[i, j] = InteractionMatrix(acc)
    end
    println("Dipolar interactions tensor constructed!")
end
