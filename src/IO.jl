using HDF5
using Serialization

function array(tuple::NTuple{N,T}) where {N,T<:Number}
    return [ x for x in tuple]
end

function writeMonteCarlo(filename::String, mc::MonteCarlo{Lattice{D,N}}) where {D,N}
    h5open(filename, "w") do f
        #write binary checkpoint
        data = IOBuffer()
        serialize(data, mc)
        f["checkpoint"] = take!(data)

        # MC parameters
        f["mc/beta"] = mc.beta
        f["mc/thermalizationSweeps"] = mc.thermalizationSweeps
        f["mc/measurementSweeps"] = mc.measurementSweeps
        f["mc/measurementRate"] = mc.measurementRate
        f["mc/reportInterval"] = mc.reportInterval
        f["mc/checkpointInterval"] = mc.checkpointInterval
        f["mc/overRelaxationRate"] = mc.overRelaxationRate
        f["mc/seed"] = mc.seed
        f["mc/sweep"] = mc.sweep

        # lattice info
        f["mc/lattice/L"] = array(mc.lattice.size)
        for i in 1:D
            f["mc/lattice/unitcell/primitive/"*string(i)] = array(mc.lattice.unitcell.primitive[i])
        end
        for i in 1:length(mc.lattice.unitcell.basis)
            f["mc/lattice/unitcell/basis/"*string(i)] = array(mc.lattice.unitcell.basis[i])
            f["mc/lattice/unitcell/interactionsOnsite/"*string(i)] = mc.lattice.unitcell.interactionsOnsite[i]
            f["mc/lattice/unitcell/interactionsField/"*string(i)] = mc.lattice.unitcell.interactionsField[i]
        end
        for i in 1:length(mc.lattice.unitcell.interactions)
            f["mc/lattice/unitcell/interactions/"*string(i)*"/b1"] = mc.lattice.unitcell.interactions[i][1]
            f["mc/lattice/unitcell/interactions/"*string(i)*"/b2"] = mc.lattice.unitcell.interactions[i][2]
            f["mc/lattice/unitcell/interactions/"*string(i)*"/offset"] = array(mc.lattice.unitcell.interactions[i][3])
            f["mc/lattice/unitcell/interactions/"*string(i)*"/M"] = mc.lattice.unitcell.interactions[i][4]
        end
        for i in 1:length(mc.lattice)
            f["mc/lattice/sitePositions/"*string(i)] = array(mc.lattice.sitePositions[i])
        end

        # observables
        saveObservables(mc.observables, f, mc)
    end
end

function readMonteCarlo(filename::String)
    h5open(filename, "r") do f
        data = IOBuffer(read(f["checkpoint"]))
        return deserialize(data)
    end
end

function saveObservables(obs::DefaultObservables, f::HDF5.File, mc::MonteCarlo{T}) where T<:Lattice
    f["mc/observables/correlation/mean"] = mean(obs.correlation)
    f["mc/observables/correlation/error"] = std_error(obs.correlation)
    f["mc/observables/correlationXY/mean"] = mean(obs.correlationXY)
    f["mc/observables/correlationXY/error"] = std_error(obs.correlationXY)
    f["mc/observables/correlationZ/mean"] = mean(obs.correlationZ)
    f["mc/observables/correlationZ/error"] = std_error(obs.correlationZ)

    ns = length(mc.lattice)
    nb = length(mc.lattice.unitcell.basis)
    β = mc.beta

    # χ = β * N * (<o²> - <o>²)
    chi(o) = β * (o[2] - o[1] * o[1]) * (ns / nb)
    ∇chi(o) = [-2.0 * β * o[1] * (ns/nb), β * (ns/nb)]
    f["mc/observables/magnetization/mean"] = means(obs.magnetization)[1]
    f["mc/observables/magnetization/error"] = std_errors(obs.magnetization)[1]
    f["mc/observables/magneticSusceptibility/mean"] = mean(obs.magnetization, chi)
    f["mc/observables/magneticSusceptibility/error"] = sqrt(abs(var(obs.magnetization, ∇chi, BinningAnalysis._reliable_level(obs.magnetization))) / obs.magnetization.count[BinningAnalysis._reliable_level(obs.magnetization)])

    # Cv = β² * (<E²> - <E>²) / N
    c(e) = β * β * (e[2] - e[1] * e[1]) * ns
    ∇c(e) = [-2.0 * β * β * e[1] * ns, β * β * ns]
    f["mc/observables/energyDensity/mean"] = means(obs.energy)[1]
    f["mc/observables/energyDensity/error"] = std_errors(obs.energy)[1]
    f["mc/observables/specificHeat/mean"] = mean(obs.energy, c)
    f["mc/observables/specificHeat/error"] = sqrt(abs(var(obs.energy, ∇c, BinningAnalysis._reliable_level(obs.energy))) / obs.energy.count[BinningAnalysis._reliable_level(obs.energy)])
end
