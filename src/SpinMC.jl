module SpinMC

include("UnitCell.jl")
export UnitCell, addInteraction!, setInteractionOnsite!, setField!, addBasisSite!
include("InteractionMatrix.jl")
include("Lattice.jl")
export Lattice, size, length, getSpin, setSpin!, getSitePosition

include("Observables.jl")
export AbstractObservables
include("Spin.jl")
export getEnergy, getMagnetization, getCorrelation, getCorrelationXY, getCorrelationZ

include("MonteCarlo.jl")
export MonteCarlo, run!

include("Helper.jl")
include("IO.jl")
export writeMonteCarlo, readMonteCarlo

using Reexport
@reexport using BinningAnalysis

end
