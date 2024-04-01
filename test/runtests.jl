using TimeResolvedDecoding
using LinearAlgebra
using StatsBase
using StableRNGs
using Test


@testset "Basic" begin 
    rng = StableRNG(1234) 
    ndims = 3
    nbins = 1
    ntrials = 100
    X = fill(0.0,ndims,nbins,ntrials)
    trialid = fill(0,ntrials)
    μ = fill(0.0, 3,2)
    μ[:,1] = [-1.0, 0.0, 1.0]
    μ[:,2] = [1.0, -1.0, -1.0]
    for i in 1:ntrials
        k = rand(rng, 1:2)
        X[:,1,i] = μ[:,k]
    end
    X .+= 0.0001*randn(rng, size(X))
    pq = TimeResolvedDecoding.decode_target_location(I(ndims), X,trialid)
    
    μ = dropdims(mean(pq.perf,dims=3),dims=3)
    @test μ ≈ [1.0;;]
end

@testset "Dynamic" begin
    rng = StableRNG(1234) 
    # Simple decoder
    ndims = 3
    nbins = 4
    ntrials = 500 
    X = fill(0.0,ndims,nbins,ntrials)
    μ = fill(0.0, 3,2)
    μ[:,1] = [-1.0, 0.0, 1.0]
    μ[:,2] = [1.0, -1.0, 1.0]
    # Dynamic code
    θ1 = π/2
    R1 = [cos(θ1) -sin(θ1) 0.0;sin(θ1) cos(θ1) 0.0; 0.0 0.0 1.0]
    θ2 = 3π/2
    R2 = [cos(θ2) -sin(θ2) 0.0;sin(θ2) cos(θ2) 0.0; 0.0 0.0 1.0]
    R3 = [cos(θ2) 0.0 -sin(θ2);0.0 0.0 1.0;sin(θ2) 0.0 cos(θ2)]
    trialid = fill(0, ntrials)
    for i in 1:ntrials
        k = rand(rng, 1:2)
        trialid[i] = k 
        X[:,1,i] = μ[:,k] 

        X[:,2,i] .= R1*X[:,1,i]
        X[:,3,i] .= R2*X[:,1,i]
        X[:,4,i] .= R3*X[:,1,i]
    end
    X .+= 0.0001*randn(rng, size(X))
    pq = TimeResolvedDecoding.decode_target_location(I(ndims), X,trialid, [[1:nbins;] for i in 1:nbins];rng=rng)
    
    μ = dropdims(mean(pq.perf,dims=3),dims=3)
    @test μ ≈ [0.95 0.5090000000000002 0.4909999999999997 0.5090000000000002;
               0.4909999999999997 0.95 0.05 0.5090000000000002;
               0.5090000000000002 0.05 0.95 0.5090000000000002;
               0.48519999999999974 0.4909999999999997 0.5090000000000002 0.95]
end