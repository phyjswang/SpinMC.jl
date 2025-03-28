using Random
using Dates
using Printf
using MPI
using TimerOutputs

mutable struct MonteCarloStatistics
    sweeps::Int
    attemptedLocalUpdates::Int
    acceptedLocalUpdates::Int
    attemptedReplicaExchanges::Int
    acceptedReplicaExchanges::Int
    initializationTime::Float64

    MonteCarloStatistics() = new(0, 0, 0, 0, 0, time())
end

mutable struct MonteCarlo{T<:Lattice,U<:AbstractRNG}
    lattice::T

    beta::Float64
    thermalizationSweeps::Int
    measurementSweeps::Int
    measurementRate::Int
    replicaExchangeRate::Int
    reportInterval::Int
    checkpointInterval::Int

    rng::U
    seed::UInt
    sweep::Int

    overRelaxationRate::Float64

    observables::AbstractObservables

    timeused::Float64
end

function MonteCarlo(
    lattice::T,
    beta::Float64,
    thermalizationSweeps::Int,
    measurementSweeps::Int;
    obs::AbstractObservables = DefaultObservables(lattice),
    measurementRate::Int = 1,
    replicaExchangeRate::Int = 10,
    reportInterval::Int = round(Int, 0.05 * (thermalizationSweeps + measurementSweeps)),
    checkpointInterval::Int = 3600,
    rng::U = copy(Random.GLOBAL_RNG),
    seed::UInt = rand(Random.RandomDevice(),UInt),
    overRelaxationRate::Float64 = 0.5,
    timeused::Float64 = 0.0
    ) where T<:Lattice where U<:AbstractRNG

    mc = MonteCarlo(
        deepcopy(lattice),
        beta,
        thermalizationSweeps,
        measurementSweeps,
        measurementRate,
        replicaExchangeRate,
        reportInterval,
        checkpointInterval,
        rng,
        seed,
        0,
        overRelaxationRate,
        obs,
        timeused
    )
    Random.seed!(mc.rng, mc.seed)

    return mc
end

@timeit_debug function overRelaxationSweep!(mc::MonteCarlo{T}) where T<:Lattice
    for _ in 1:length(mc.lattice)
        # select random spin
        site = rand(mc.rng, 1:length(mc.lattice))
        oldState = getSpin(mc.lattice, site)

        # calculate local field only
        localField = calLocalField(mc.lattice, site)

        if norm(localField) > 1e-8 # avoid division by zero
            # component parallel to the local field
            spinpara = (dot(oldState, localField) / norm(localField)^2 ) .* localField

            # reflect across the local field
            newState = 2 .* spinpara .- oldState
        else
            # if the local field is zero, use a random spin
            newState = uniformOnSphere(mc.rng)
        end
        setSpin!(mc.lattice, site, newState)
    end
end

@timeit_debug function metropolisSweep!(mc::MonteCarlo{T}, statistics::MonteCarloStatistics, energy::Float64) where T<:Lattice
    for _ in 1:length(mc.lattice)
        #select random spin
        site = rand(mc.rng, 1:length(mc.lattice))

        #propose new spin configuration
        newSpinState = uniformOnSphere(mc.rng)
        energyDifference = getEnergyDifference(mc.lattice, site, newSpinState)

        #check acceptance of new configuration
        statistics.attemptedLocalUpdates += 1
        p = exp(-mc.beta * energyDifference)
        if (rand(mc.rng) < min(1.0, p))
            ds = newSpinState .- getSpin(mc.lattice, site)
            # update local fields
            interactionSites = getInteractionSites(mc.lattice, site)
            if mc.lattice.unitcell.dipolar ≠ 0.0
                for i in vcat(1:site-1,site+1:length(mc.lattice))
                    if i ∈ interactionSites
                        updateLocalField_complex!(mc.lattice,  i, site, ds)
                    else
                        updateLocalField_dipolar!(mc.lattice,  i, site, ds)
                    end
                end
            else
                for i in eachindex(interactionSites)
                    updateLocalField_simple!(mc.lattice, interactionSites[i], site, ds)
                end
            end
            # update spin
            setSpin!(mc.lattice, site, newSpinState)
            energy += energyDifference
            statistics.acceptedLocalUpdates += 1
        end
    end
    return energy
end

function run!(
    mc::MonteCarlo{T};
    outfile::Union{String,Nothing}=nothing,
    timer::Bool = false,
    isquiet::Bool = false,
) where T<:Lattice
    timer && reset_timer!()

    #init MPI
    rank = 0
    commSize = 1
    allBetas = zeros(0)
    enableMPI = false
    if MPI.Initialized()
        commSize = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if commSize > 1
            allBetas = zeros(commSize)
            allBetas[rank + 1] = mc.beta
            MPI.Allgather!(UBuffer(allBetas, 1), MPI.COMM_WORLD)
            enableMPI = true
            rank == 0 && !isquiet && @printf("MPI detected. Enabling replica exchanges across %d simulations.\n", commSize)
        end
    end

    #init IO
    enableOutput = typeof(outfile) != Nothing
    if enableOutput
        isfile(outfile) && error("File ", outfile, " already exists. Terminating.")
    end

    # initialization
    if mc.sweep == 0
        #init spin configuration
        for i in 1:length(mc.lattice)
            setSpin!(mc.lattice, i, uniformOnSphere(mc.rng))
        end
        # init local fields
        for i in 1:length(mc.lattice)
            setLocalField!(mc.lattice, i, calLocalField(mc.lattice, i))
        end
    end

    #init Monte Carlo run
    totalSweeps = mc.thermalizationSweeps + mc.measurementSweeps
    partnerSpinConfiguration = deepcopy(mc.lattice.spins)
    partnerLocalFields = deepcopy(mc.lattice.localFields)
    energy = getEnergy(mc.lattice)

    #launch Monte Carlo run
    lastCheckpointTime = time()
    statistics = MonteCarloStatistics()
    rank == 0 && !isquiet && @printf("Simulation started on %s.\n\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    # perform over-relaxation only if there is no onsite interactions
    allowOverRelaxation = maximum(maximum.(mc.lattice.unitcell.interactionsOnsite)) == 0.0 && mc.lattice.unitcell.dipolar == 0.0

    timeused = @elapsed while mc.sweep < totalSweeps
        # perform over-relaxation step
        if allowOverRelaxation && mc.overRelaxationRate > 0.0
            if mc.overRelaxationRate < 1.0
                if rand(mc.rng) < mc.overRelaxationRate
                    overRelaxationSweep!(mc)
                end
            else
                for _ in 1:mc.overRelaxationRate
                    overRelaxationSweep!(mc)
                end
            end
            # update local fields after over-relaxation
            for i in 1:length(mc.lattice)
                setLocalField!(mc.lattice, i, calLocalField(mc.lattice, i))
            end
        end

        #perform local sweep
        energy = metropolisSweep!(mc, statistics, energy)
        statistics.sweeps += 1

        #perform replica exchange
        if enableMPI && mc.sweep % mc.replicaExchangeRate == 0
            #determine MPI rank to exchange configuration with
            if iseven(mc.sweep ÷ mc.replicaExchangeRate)
                partnerRank = iseven(rank) ? rank + 1 : rank - 1
            else
                partnerRank = iseven(rank) ? rank - 1 : rank + 1
            end

            if partnerRank >= 0 && partnerRank < commSize
                #obtain energy of new configuration
                partnerEnergy = MPISendrecvFloat(energy, partnerRank, MPI.COMM_WORLD)

                #check acceptance of new configuration
                statistics.attemptedReplicaExchanges += 1
                exchangeAccepted = false
                if iseven(rank)
                    p = exp(-(allBetas[rank + 1] - allBetas[partnerRank + 1]) * (partnerEnergy - energy))
                    exchangeAccepted = (rand(mc.rng) < min(1.0, p))
                    MPISendBool(exchangeAccepted, partnerRank, MPI.COMM_WORLD)
                else
                    exchangeAccepted = MPIRecvBool(partnerRank, MPI.COMM_WORLD)
                end
                if (exchangeAccepted)
                    energy = partnerEnergy
                    MPI.Sendrecv!(mc.lattice.spins, partnerRank, 0, partnerSpinConfiguration, partnerRank, 0, MPI.COMM_WORLD)
                    (mc.lattice.spins, partnerSpinConfiguration) = (partnerSpinConfiguration, mc.lattice.spins)
                    MPI.Sendrecv!(mc.lattice.localFields, partnerRank, 0, partnerLocalFields, partnerRank, 0, MPI.COMM_WORLD)
                    (mc.lattice.localFields, partnerLocalFields) = (partnerLocalFields, mc.lattice.localFields)
                    statistics.acceptedReplicaExchanges += 1
                end
            end
        end

        #perform measurement
        if mc.sweep >= mc.thermalizationSweeps
            if mc.sweep % mc.measurementRate == 0
                performMeasurements!(mc.observables, mc.lattice, energy)
            end
        end

        #increment sweep
        statistics.sweeps += 1
        mc.sweep += 1

        #runtime statistics
        t = time()
        if mc.sweep % mc.reportInterval == 0
            #collect statistics
            progress = 100.0 * mc.sweep / totalSweeps
            thermalized = (mc.sweep >= mc.thermalizationSweeps) ? "YES" : "NO"
            sweeprate = statistics.sweeps / (t - statistics.initializationTime)
            sweeptime = 1.0 / sweeprate
            eta = (totalSweeps - mc.sweep) / sweeprate

            localUpdateAcceptanceRate = 100.0 * statistics.acceptedLocalUpdates / statistics.attemptedLocalUpdates
            if enableMPI
                replicaExchangeAcceptanceRate = 100.0 * statistics.acceptedReplicaExchanges / statistics.attemptedReplicaExchanges
                allLocalAppectanceRate = zeros(commSize)
                allLocalAppectanceRate[rank + 1] = localUpdateAcceptanceRate
                MPI.Allgather!(UBuffer(allLocalAppectanceRate, 1), MPI.COMM_WORLD)
                allReplicaExchangeAcceptanceRate = zeros(commSize)
                allReplicaExchangeAcceptanceRate[rank + 1] = replicaExchangeAcceptanceRate
                MPI.Allgather!(UBuffer(allReplicaExchangeAcceptanceRate, 1), MPI.COMM_WORLD)
            end

            #print statistics
            if (rank == 0) && !isquiet
                str = ""
                str *= @sprintf("Sweep %d / %d (%.1f%%)", mc.sweep, totalSweeps, progress)
                str *= @sprintf("\t\tETA : %s\n", Dates.format(Dates.now() + Dates.Second(round(Int64,eta)), "dd u yyyy HH:MM:SS"))
                str *= @sprintf("\t\tthermalized : %s\n", thermalized)
                str *= @sprintf("\t\tsweep rate : %.1f sweeps/s\n", sweeprate)
                str *= @sprintf("\t\tsweep duration : %.3f ms\n", sweeptime * 1000)

                if enableMPI
                    for n in 1:commSize
                        str *= @sprintf("\t\tsimulation %d update acceptance rate: %.2f%%\n", n - 1, allLocalAppectanceRate[n])
                        str *= @sprintf("\t\tsimulation %d replica exchange acceptance rate : %.2f%%\n", n - 1, allReplicaExchangeAcceptanceRate[n])
                    end
                else
                    str *= @sprintf("\t\tupdate acceptance rate: %.2f%%\n", localUpdateAcceptanceRate)
                end
                str *= @sprintf("\n")
                print(str)
            end

            #reset statistics
            statistics = MonteCarloStatistics()
            timer && print_timer()
        end

        #write checkpoint
        if enableOutput
            checkpointPending = time() - lastCheckpointTime >= mc.checkpointInterval
            enableMPI && (checkpointPending = MPIBcastBool(checkpointPending, 0, MPI.COMM_WORLD))
            if checkpointPending
                writeMonteCarlo(outfile, mc)
                lastCheckpointTime = time()
                rank == 0 && !isquiet && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
            end
        end
        flush(stdout)
    end

    mc.timeused += timeused

    #write final checkpoint
    if enableOutput
        writeMonteCarlo(outfile, mc)
        rank == 0 && !isquiet && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    end

    #return
    rank == 0 && !isquiet && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    return nothing
end

function runSA!(
    mc::MonteCarlo{T},
    changeRate::Float64;
    outfile::Union{String,Nothing}=nothing,
    timer::Bool = false,
    isquiet::Bool = false,
) where T<:Lattice
    timer && reset_timer!()

    #init IO
    enableOutput = typeof(outfile) != Nothing
    if enableOutput
        isfile(outfile) && error("File ", outfile, " already exists. Terminating.")
    end

    # initialization
    if mc.sweep == 0
        #init spin configuration
        for i in 1:length(mc.lattice)
            setSpin!(mc.lattice, i, uniformOnSphere(mc.rng))
        end
        # init local fields
        for i in 1:length(mc.lattice)
            setLocalField!(mc.lattice, i, calLocalField(mc.lattice, i))
        end
    end

    #init Monte Carlo run
    totalSweeps = mc.thermalizationSweeps + mc.measurementSweeps
    energy = getEnergy(mc.lattice)

    #launch Monte Carlo run
    lastCheckpointTime = time()
    statistics = MonteCarloStatistics()
    !isquiet && @printf("Simulation started on %s.\n\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    # perform over-relaxation only if there is no onsite interactions
    allowOverRelaxation = maximum(maximum.(mc.lattice.unitcell.interactionsOnsite)) == 0.0 && mc.lattice.unitcell.dipolar == 0.0

    timeused = @elapsed while mc.sweep < totalSweeps
        # TODO over-relaxation
        # # perform over-relaxation step
        # if allowOverRelaxation && mc.overRelaxationRate > 0.0
        #     if mc.overRelaxationRate < 1.0
        #         if rand(mc.rng) < mc.overRelaxationRate
        #             overRelaxationSweep!(mc)
        #         end
        #     else
        #         for _ in 1:mc.overRelaxationRate
        #             overRelaxationSweep!(mc)
        #         end
        #     end
        #     # update local fields after over-relaxation
        #     for i in 1:length(mc.lattice)
        #         setLocalField!(mc.lattice, i, calLocalField(mc.lattice, i))
        #     end
        # end

        #perform local sweep
        if mc.sweep ≤ mc.thermalizationSweeps
            mc.beta *= changeRate^-1
        end
        energy = metropolisSweep!(mc, statistics, energy)
        statistics.sweeps += 1

        #perform measurement
        if mc.sweep >= mc.thermalizationSweeps
            if mc.sweep % mc.measurementRate == 0
                performMeasurements!(mc.observables, mc.lattice, energy)
            end
        end

        #increment sweep
        statistics.sweeps += 1
        mc.sweep += 1

        #runtime statistics
        t = time()
        if mc.sweep % mc.reportInterval == 0
            #collect statistics
            progress = 100.0 * mc.sweep / totalSweeps
            thermalized = (mc.sweep >= mc.thermalizationSweeps) ? "YES" : "NO"
            sweeprate = statistics.sweeps / (t - statistics.initializationTime)
            sweeptime = 1.0 / sweeprate
            eta = (totalSweeps - mc.sweep) / sweeprate

            localUpdateAcceptanceRate = 100.0 * statistics.acceptedLocalUpdates / statistics.attemptedLocalUpdates

            #print statistics
            if !isquiet
                str = ""
                str *= @sprintf("Sweep %d / %d (%.1f%%)", mc.sweep, totalSweeps, progress)
                str *= @sprintf("\t\tETA : %s\n", Dates.format(Dates.now() + Dates.Second(round(Int64,eta)), "dd u yyyy HH:MM:SS"))
                str *= @sprintf("\t\tthermalized : %s\n", thermalized)
                str *= @sprintf("\t\tsweep rate : %.1f sweeps/s\n", sweeprate)
                str *= @sprintf("\t\tsweep duration : %.3f ms\n", sweeptime * 1000)

                str *= @sprintf("\t\tupdate acceptance rate: %.2f%%\n", localUpdateAcceptanceRate)
                str *= @sprintf("\n")
                print(str)
            end

            #reset statistics
            statistics = MonteCarloStatistics()
            timer && print_timer()
        end

        #write checkpoint
        if enableOutput
            checkpointPending = time() - lastCheckpointTime >= mc.checkpointInterval
            if checkpointPending
                writeMonteCarlo(outfile, mc)
                lastCheckpointTime = time()
                !isquiet && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
            end
        end
        flush(stdout)
    end

    mc.timeused += timeused

    #write final checkpoint
    if enableOutput
        writeMonteCarlo(outfile, mc)
        !isquiet && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    end

    #return
    !isquiet && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    return nothing
end
