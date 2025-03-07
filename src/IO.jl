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
        β = mc.beta
        f["mc/beta"] = β
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
        f["mc/observables/correlation/mean"] = mean(mc.observables.correlation)
        f["mc/observables/correlation/error"] = std_error(mc.observables.correlation)
        f["mc/observables/correlationXY/mean"] = mean(mc.observables.correlationXY)
        f["mc/observables/correlationXY/error"] = std_error(mc.observables.correlationXY)
        f["mc/observables/correlationZ/mean"] = mean(mc.observables.correlationZ)
        f["mc/observables/correlationZ/error"] = std_error(mc.observables.correlationZ)
        f["mc/observables/phi/mean"] = mean(mc.observables.phi)
        f["mc/observables/phi/error"] = std_error(mc.observables.phi)

        ns = length(mc.lattice)
        nb = length(mc.lattice.unitcell.basis)

        # χ = β * N * (<o²> - <o>²)
        chi(o) = β * (o[2] - o[1] * o[1]) * (ns / nb)
        ∇chi(o) = [-2.0 * β * o[1] * (ns/nb), β * (ns/nb)]
        f["mc/observables/magnetization/mean"] = means(mc.observables.magnetization)[1]
        f["mc/observables/magnetization/error"] = std_errors(mc.observables.magnetization)[1]
        f["mc/observables/magneticSusceptibility/mean"] = mean(mc.observables.magnetization, chi)
        f["mc/observables/magneticSusceptibility/error"] = sqrt(abs(var(mc.observables.magnetization, ∇chi, BinningAnalysis._reliable_level(mc.observables.magnetization))) / mc.observables.magnetization.count[BinningAnalysis._reliable_level(mc.observables.magnetization)])
        f["mc/observables/afpara/mean"] = means(mc.observables.afpara)[1]
        f["mc/observables/afpara/error"] = std_errors(mc.observables.afpara)[1]
        f["mc/observables/afparaSusceptibility/mean"] = mean(mc.observables.afpara, chi)
        f["mc/observables/afparaSusceptibility/error"] = sqrt(abs(var(mc.observables.afpara, ∇chi, BinningAnalysis._reliable_level(mc.observables.afpara))) / mc.observables.afpara.count[BinningAnalysis._reliable_level(mc.observables.afpara)])
        f["mc/observables/aaperp/mean"] = means(mc.observables.aaperp)[1]
        f["mc/observables/aaperp/error"] = std_errors(mc.observables.aaperp)[1]
        f["mc/observables/aaperpSusceptibility/mean"] = mean(mc.observables.aaperp, chi)
        f["mc/observables/aaperpSusceptibility/error"] = sqrt(abs(var(mc.observables.aaperp, ∇chi, BinningAnalysis._reliable_level(mc.observables.aaperp))) / mc.observables.aaperp.count[BinningAnalysis._reliable_level(mc.observables.aaperp)])

        # Cv = β² * (<E²> - <E>²) / N
        c(e) = β * β * (e[2] - e[1] * e[1]) * ns
        ∇c(e) = [-2.0 * β * β * e[1] * ns, β * β * ns]
        f["mc/observables/specificHeat/mean"] = mean(mc.observables.energy, c)
        f["mc/observables/specificHeat/error"] = sqrt(abs(var(mc.observables.energy, ∇c, BinningAnalysis._reliable_level(mc.observables.energy))) / mc.observables.energy.count[BinningAnalysis._reliable_level(mc.observables.energy)])

        f["mc/observables/timeused"] = mc.observables.timeused
    end
end

function readMonteCarlo(filename::String)
    h5open(filename, "r") do f
        data = IOBuffer(read(f["checkpoint"]))
        return deserialize(data)
    end
end
