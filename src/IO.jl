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

        #write human readable results and parameters
        f["mc/beta"] = mc.beta
        f["mc/thermalizationSweeps"] = mc.thermalizationSweeps
        f["mc/measurementSweeps"] = mc.measurementSweeps
        f["mc/measurementRate"] = mc.measurementRate
        f["mc/reportInterval"] = mc.reportInterval
        f["mc/checkpointInterval"] = mc.checkpointInterval
        f["mc/overRelaxationRate"] = mc.overRelaxationRate
        f["mc/seed"] = mc.seed
        f["mc/sweep"] = mc.sweep

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

        f["mc/observables/energyDensity/mean"] = means(mc.observables.energy)[1]
        f["mc/observables/energyDensity/error"] = std_errors(mc.observables.energy)[1]
        f["mc/observables/magnetizationVector/mean"] = mean(mc.observables.lsmagnetizationVector)
        f["mc/observables/magnetizationVector/error"] = std_error(mc.observables.lsmagnetizationVector)
        f["mc/observables/correlation/mean"] = mean(mc.observables.correlation)
        f["mc/observables/correlation/error"] = std_error(mc.observables.correlation)
        f["mc/observables/correlationXY/mean"] = mean(mc.observables.correlationXY)
        f["mc/observables/correlationXY/error"] = std_error(mc.observables.correlationXY)
        f["mc/observables/correlationZ/mean"] = mean(mc.observables.correlationZ)
        f["mc/observables/correlationZ/error"] = std_error(mc.observables.correlationZ)

        N = length(mc.lattice)
        nb = length(mc.lattice.basis)

        # χ = β * N * (<o²> - <o>²)
        chi(o) = mc.beta * (o[2] - o[1] * o[1]) * (N / nb)
        ∇chi(o) = [-2.0 * mc.beta * o[1] * (N/nb), mc.beta * (N/nb)]
        f["mc/observables/afpara/mean"] = mean(mc.observables.afpara)
        f["mc/observables/afpara/error"] = std_error(mc.observables.afpara)
        f["mc/observables/afparaSusceptibility/mean"] = mean(mc.observables.afpara, chi)
        f["mc/observables/afparaSusceptibility/error"] = sqrt(abs(var(mc.observables.afpara, ∇chi, BinningAnalysis._reliable_level(mc.observables.afpara))) / mc.observables.afpara.count[BinningAnalysis._reliable_level(mc.observables.afpara)])
        f["mc/observables/aaperp/mean"] = mean(mc.observables.aaperp)
        f["mc/observables/aaperp/error"] = std_error(mc.observables.aaperp)
        f["mc/observables/aaperpSusceptibility/mean"] = mean(mc.observables.aaperp)
        f["mc/observables/aaperpSusceptibility/error"] = sqrt(abs(var(mc.observables.aaperp, ∇chi, BinningAnalysis._reliable_level(mc.observables.aaperp))) / mc.observables.aaperp.count[BinningAnalysis._reliable_level(mc.observables.aaperp)])

        # Cv = β² * (<E²> - <E>²) / N
        c(e) = mc.beta * mc.beta * (e[2] - e[1] * e[1]) * length(mc.lattice)
        ∇c(e) = [-2.0 * mc.beta * mc.beta * e[1] * length(mc.lattice), mc.beta * mc.beta * length(mc.lattice)]
        heat = mean(mc.observables.energy, c)
        dheat = sqrt(abs(var(mc.observables.energy, ∇c, BinningAnalysis._reliable_level(mc.observables.energy))) / mc.observables.energy.count[BinningAnalysis._reliable_level(mc.observables.energy)])
        f["mc/observables/specificHeat/mean"] = heat
        f["mc/observables/specificHeat/error"] = dheat
    end
end

function readMonteCarlo(filename::String)
    h5open(filename, "r") do f
        data = IOBuffer(read(f["checkpoint"]))
        return deserialize(data)
    end
end
