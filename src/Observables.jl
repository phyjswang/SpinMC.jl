using BinningAnalysis
using TimerOutputs

abstract type AbstractObservables end

mutable struct DefaultObservables <: AbstractObservables
    energy::ErrorPropagator{Float64,32}
    magnetization::ErrorPropagator{Float64,32}
    correlation::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    correlationXY::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
    correlationZ::LogBinner{Array{Float64,2},32,BinningAnalysis.Variance{Array{Float64,2}}}
end

function DefaultObservables(lattice::T) where T<:Lattice
    return DefaultObservables(
        ErrorPropagator(Float64),
        ErrorPropagator(Float64),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
        LogBinner(zeros(Float64,lattice.length, length(lattice.unitcell.basis))),
    )
end

function performMeasurements!(observables::DefaultObservables, lattice::T, energy::Float64) where T<:Lattice
    #measure energy and energy^2
    push!(observables.energy, energy / length(lattice), energy * energy / (length(lattice) * length(lattice)))

    #measure magnetization
    lsm = getMagnetization(lattice)
    m = sum(reshape(lsm,3,:),dims=2) / length(lattice.unit.basis)
    push!(observables.magnetization, norm(m), norm(m)*norm(m))

    #measure spin correlations
    push!(observables.correlation, getCorrelation(lattice))
    push!(observables.correlationXY, getCorrelationXY(lattice))
    push!(observables.correlationZ, getCorrelationZ(lattice))
end
