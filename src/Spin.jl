using Random
using LinearAlgebra

function uniformOnSphere(rng = Random.GLOBAL_RNG)::NTuple{3,Float64}
    phi = 2.0 * pi * rand(rng)
    z = 2.0 * rand(rng) - 1.0;
    r = sqrt(1.0 - z * z)
    return (r * cos(phi), r * sin(phi), z)
end

function exchangeEnergy(s1::Tuple{Float64,Float64,Float64}, M::InteractionMatrix, s2::Tuple{Float64,Float64,Float64})::Float64
    return s1[1] * (M.m11 * s2[1] + M.m12 * s2[2] + M.m13 * s2[3]) + s1[2] * (M.m21 * s2[1] + M.m22 * s2[2] + M.m23 * s2[3]) + s1[3] * (M.m31 * s2[1] + M.m32 * s2[2] + M.m33 * s2[3])
end

function localFieldFromExchangeEnergy(M::InteractionMatrix, s2::Tuple{Float64,Float64,Float64})::NTuple{3,Float64}
    return ((M.m11 * s2[1] + M.m12 * s2[2] + M.m13 * s2[3]), (M.m21 * s2[1] + M.m22 * s2[2] + M.m23 * s2[3]), (M.m31 * s2[1] + M.m32 * s2[2] + M.m33 * s2[3]))
end

function getEnergy_old(lattice::Lattice{D,N})::Float64 where {D,N}
    energy = 0.0

    for site in 1:length(lattice)
        s0 = getSpin(lattice, site)

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        for i in 1:length(interactionSites)
            if site > interactionSites[i] # avoid double counting
                energy += exchangeEnergy(s0, getInteractionMatrices(lattice, site,i), getSpin(lattice, interactionSites[i]))
            end
        end

        #onsite interaction
        energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        energy += dot(s0, getInteractionField(lattice, site))
    end

    return energy
end


# definition of local field is
# Dᵢ = Cᵢ + ∑ⱼ AᵢⱼSⱼ (j ≠ i)
# On-site interactions are excluded, since they depend on the local spin itself, they should be dealt with separately
@timeit_debug function calLocalField(lattice::Lattice{D,N}, site::Int)::Tuple{Float64,Float64,Float64} where {D,N}
    hx, hy, hz = 0.0, 0.0, 0.0

    # two-spin interactions
    interactionSites = getInteractionSites(lattice, site)
    for i in eachindex(interactionSites)
        localField = localFieldFromExchangeEnergy(getInteractionMatrices(lattice, site, i), getSpin(lattice, interactionSites[i]))
        hx += localField[1]
        hy += localField[2]
        hz += localField[3]
    end

    # onsite field interaction
    localField = getInteractionField(lattice, site)
    hx += localField[1]
    hy += localField[2]
    hz += localField[3]

    # dipolar contribution
    if lattice.unitcell.dipolar ≠ 0.0
        for i in vcat(1:site-1,site+1:lattice.length)
            localField = localFieldFromExchangeEnergy(getInteractionDipolar(lattice, site, i), getSpin(lattice, i))
            hx += localField[1] * lattice.unitcell.dipolar
            hy += localField[2] * lattice.unitcell.dipolar
            hz += localField[3] * lattice.unitcell.dipolar
        end
    end

    return (hx,hy,hz)
end

# total energy is E = ∑ᵢ HᵢSᵢ
# Hᵢ = ∑ⱼ AᵢⱼSⱼ / 2 + BᵢSᵢ + Cᵢ
#    = BᵢSᵢ + (Cᵢ + Dᵢ) / 2
function getEnergy(lattice::Lattice{D,N})::Float64 where {D,N}
    energy = 0.0
    if lattice.unitcell.dipolar == 0
        for site in 1:length(lattice)
            s0 = getSpin(lattice, site)
            # 1/2 to avoid double counting
            energy += dot(
                s0,
                (getLocalField(lattice, site) .+ getInteractionField(lattice, site)) ./ 2 .+
                localFieldFromExchangeEnergy(getInteractionOnsite(lattice, site), s0)
            )
        end
    else
        for site in 1:length(lattice)
            s0 = getSpin(lattice, site)
            # 1/2 to avoid double counting
            energy += dot(
                s0,
                (getLocalField(lattice, site) .+ getInteractionField(lattice, site)) ./ 2 .+
                localFieldFromExchangeEnergy(getInteractionOnsite(lattice, site), s0) .+
                localFieldFromExchangeEnergy(getInteractionDipolar(lattice, site, site), s0) .* lattice.unitcell.dipolar
            )
        end
    end
    return energy
end

# function getEnergyDifference(lattice::Lattice{D,N}, site::Int, newState::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
#     dE = 0.0
#     oldState = getSpin(lattice, site)
#     ds = newState .- oldState

#     #two-spin interactions
#     interactionSites = getInteractionSites(lattice, site)
#     interactionMatrices = getInteractionMatrices(lattice, site)
#     for i in 1:length(interactionSites)
#         dE += exchangeEnergy(ds, interactionMatrices[i], getSpin(lattice, interactionSites[i]))
#     end

#     #onsite interaction
#     interactionOnsite = getInteractionOnsite(lattice, site)
#     dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

#     #field interaction
#     dE += dot(ds, getInteractionField(lattice, site))

#     return dE
# end

# energy difference contains two terms
# ΔE = (S'ᵢBᵢS'ᵢ - SᵢBᵢSᵢ) + Dᵢ ΔSᵢ
function getEnergyDifference(lattice::Lattice{D,N}, site::Int, newState::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    oldState = getSpin(lattice, site)
    # by definition
    dE = dot(newState .- oldState, getLocalField(lattice, site))

    #onsite interaction difference
    interactionOnsite = getInteractionOnsite(lattice, site)
    dE += exchangeEnergy(newState, interactionOnsite, newState) - exchangeEnergy(oldState, interactionOnsite, oldState)

    # onsite energy difference due to dipolar interaction
    if lattice.unitcell.dipolar ≠ 0.0
        interactionOnsiteDipolar = getInteractionDipolar(lattice, site, site)
        dE += (exchangeEnergy(newState, interactionOnsiteDipolar, newState) - exchangeEnergy(oldState, interactionOnsiteDipolar, oldState)) * lattice.unitcell.dipolar
    end

    return dE
end

# change of local field at site j due to change of spin at site i
# Dⱼ → D'ⱼ = Dⱼ + Aⱼᵢ dSᵢ
# simple means no dipolar interaction
@timeit_debug function updateLocalField_simple!(lattice::Lattice{D,N}, sitej::Int, site::Int, ds::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    interactionSites = getInteractionSites(lattice, sitej)
    idx = findfirst(x -> x == site, interactionSites)
    setLocalField!(
        lattice,
        sitej,
        getLocalField(lattice, sitej) .+
        localFieldFromExchangeEnergy(getInteractionMatrices(lattice, sitej, idx), ds)
    )
end

# complex means dipolar interaction is included
@timeit_debug function updateLocalField_complex!(lattice::Lattice{D,N}, sitej::Int, site::Int, ds::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    interactionSites = getInteractionSites(lattice, sitej)
    idx = findfirst(x -> x == site, interactionSites)
    setLocalField!(
        lattice,
        sitej,
        getLocalField(lattice, sitej) .+
        localFieldFromExchangeEnergy(getInteractionMatrices(lattice, sitej, idx), ds) .+
        localFieldFromExchangeEnergy(getInteractionDipolar(lattice, sitej, site), ds) .* lattice.unitcell.dipolar
    )
end

@timeit_debug function updateLocalField_dipolar!(lattice::Lattice{D,N}, sitej::Int, site::Int, ds::Tuple{Float64,Float64,Float64})::Float64 where {D,N}
    setLocalField!(
        lattice,
        sitej,
        getLocalField(lattice, sitej) .+
        localFieldFromExchangeEnergy(getInteractionDipolar(lattice, sitej, site), ds) .* lattice.unitcell.dipolar
    )
end

function getMagnetization_old(lattice::Lattice{D,N}) where {D,N}
    mx, my, mz = 0.0, 0.0, 0.0
    for i in 1:length(lattice)
        spin = getSpin(lattice, i)
        mx += spin[1]
        my += spin[2]
        mz += spin[3]
    end
    return [mx, my, mz] / length(lattice)
end

# sublattice dependent magnetization
# lsm = [m1x, m1y, m1z, m2x, m2y, m2z, ...]
function getMagnetization(lattice::Lattice{D,N}) where {D,N}
    nb = length(lattice.unitcell.basis)
    lsm = zeros(3*nb)
    for i in 1:length(lattice)
        spin = getSpin(lattice, i)
        lsm[(mod1(i,nb)-1)*3+1] += spin[1]
        lsm[(mod1(i,nb)-1)*3+2] += spin[2]
        lsm[(mod1(i,nb)-1)*3+3] += spin[3]
    end
    return lsm ./ (length(lattice)/nb)
end

function getCorrelation(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice.unitcell.basis))
    for i in 1:length(lattice.unitcell.basis)
        s0 = getSpin(lattice, i)
        for j in 1:length(lattice)
            corr[j,i] = dot(s0, getSpin(lattice, j))
        end
    end
    return corr
end

function getCorrelationFull(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice))
    for i in 1:length(lattice)
        s0 = getSpin(lattice, i)
        for j in i:length(lattice)
            corr[j,i] = dot(s0, getSpin(lattice, j))
            corr[i,j] = corr[j,i]
        end
    end
    return corr
end

function getCorrelationXY(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice.unitcell.basis))
    for i in 1:length(lattice.unitcell.basis)
        s0 = getSpin(lattice, i)
        for j in 1:length(lattice)
            corr[j,i] = dot(s0[1:2], getSpin(lattice, j)[1:2])
        end
    end
    return corr
end

function getCorrelationFullXY(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice))
    for i in 1:length(lattice)
        s0 = getSpin(lattice, i)
        for j in i:length(lattice)
            corr[j,i] = dot(s0[1:2], getSpin(lattice, j)[1:2])
            corr[i,j] = corr[j,i]
        end
    end
    return corr
end

function getCorrelationZ(lattice::Lattice{D,N}) where {D,N}
    corr = zeros(length(lattice), length(lattice.unitcell.basis))
    for i in 1:length(lattice.unitcell.basis)
        s0 = getSpin(lattice, i)
        for j in 1:length(lattice)
            corr[j,i] = s0[3] * getSpin(lattice, j)[3]
        end
    end
    return corr
end
