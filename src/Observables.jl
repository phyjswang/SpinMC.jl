using BinningAnalysis
using TimerOutputs

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::ErrorPropagator{Float64,32}
    lsmagnetizationVector::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    afpara::ErrorPropagator{Float64,32}
    aaperp::ErrorPropagator{Float64,32}
    correlation::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    correlationXY::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    correlationZ::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    timeused::Float64
end

function Observables(lattice::T) where T<:Lattice
    return Observables(
        ErrorPropagator(Float64),
        ErrorPropagator(Float64),
        LogBinner(zeros(Float64,12)),
        ErrorPropagator(Float64),
        ErrorPropagator(Float64),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
        0.0
    )
end

@timeit_debug function performMeasurements!(observables::Observables, lattice::T, energy::Float64) where T<:Lattice
    #measure energy and energy^2
    push!(observables.energy, energy / length(lattice), energy * energy / (length(lattice) * length(lattice)))

    #measure magnetization
    lsm = getMagnetization(lattice)
    push!(observables.lsmagnetizationVector, lsm)

    m = sum(reshape(lsm,3,:),dims=2)
    push!(observables.magnetization, norm(m), norm(m)*norm(m))

    afpara = √((lsm[1] + lsm[1+3] - lsm[1+6] - lsm[1+9])^2 + (lsm[2] + lsm[2+6] - lsm[2+3] - lsm[2+9])^2) / 4
    push!(observables.afpara, afpara, afpara * afpara)
    aaperp = (abs(lsm[3]) + abs(lsm[3+3]) + abs(lsm[3+6]) + abs(lsm[3+9])) / 4
    push!(observables.aaperp, aaperp, aaperp * aaperp)

    #measure spin correlations
    push!(observables.correlation, getCorrelation(lattice))
    push!(observables.correlationXY, getCorrelationXY(lattice))
    push!(observables.correlationZ, getCorrelationZ(lattice))
end
