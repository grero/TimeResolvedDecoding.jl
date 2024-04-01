module TimeResolvedDecoding
using StatsBase
using LinearAlgebra
using Random
using MultivariateStats

IdentityTransform = Diagonal{Bool, Vector{Bool}}

struct TimeResolvedDecoder{T1<:Union{MultivariateStats.AbstractDimensionalityReduction,IdentityTransform}, T2<:MultivariateStats.RegressionModel}
    fa::T1
    testbins::Vector{Vector{Int64}}
    decoder::Vector{T2}
    perf::Array{Float64,3}
end

function decode_target_location(X::Array{T,3}, trialid;maxoutdim=1) where T <: Real
    fa = fit(FactorAnalysis, X;maxoutdim=maxoutdim)
    decode_target_location(fa, X, trialid)
end

StatsBase.predict(m::IdentityTransform, X) where T <: Real = X
MultivariateStats.outdim(m::IdentityTransform) = size(m,2)

function decode_target_location(fa, X::Array{T,3}, trialid,testbins::Union{Nothing,Vector{Vector{Int64}}}, trial_outcome;kvs...) where T <: Real
    cidx=trial_outcome.==1
    decode_target_location(fa, X[:,:,cidx], trialid[cidx],testbins;kvs...)
end 

"""
    decode_target_location(fa, X::Array{T,3}, trialid,testbins::Union{Nothing,Vector{Vector{Int64}}};nruns=100) where T <: Real

Train a decoder to decode the trial labels in `trialid`. If `testbins` is nothing, use the same bins for training and testing. If not, 
`testbin` indicates which bins to use for testing for each training bin, such that `length(testbins) == size(X,3)`.
"""
function decode_target_location(fa, X::Array{T,3}, trialid,testbins::Union{Nothing,Vector{Vector{Int64}}}=nothing;rng=Random.default_rng(),nruns=100) where T <: Real
    ndims, nbins,ntrials = size(X)
    ndims = outdim(fa)
    if testbins === nothing
        testbins = [[i] for i in 1:nbins]
    end
    nn = length.(testbins)
    ntrain = maximum(nn)
    mintrain = minimum(minimum.(testbins))
    maxtrain = maximum(maximum.(testbins))
    if !isa(fa, Diagonal{Bool, Vector{Bool}})
        Y = fill(0.0, ndims, nbins, ntrials)
        for j in 1:bins
            Y[:,j,:] = predict(fa, X[:,j,:])
        end
    else
        Y = X
    end
    label = MultivariateStats.toindices(trialid)

    perf = fill(0.0, nbins, ntrain,nruns)
    lda = Vector{MulticlassLDA{T}}(undef, maxtrain-mintrain+1)
    _ntrain = round(Int64, 0.8*ntrials)
    for r in 1:nruns
        # train
        trainidx = shuffle(rng,1:ntrials)[1:_ntrain]
        sort!(trainidx)
        testidx = setdiff(1:ntrials, trainidx)
        _ntest = length(testidx)
        trainlabel = label[trainidx]
        testlabel = label[testidx]
        for (jj,j) in enumerate(mintrain:maxtrain)
            lda[jj] = fit(MulticlassLDA, Y[:,j,trainidx], trainlabel)
        end
        # test
        for j in 1:nbins 
            y = Y[:,j,testidx]
            for (kk,k) in enumerate(testbins[j])
                Zp = predict(lda[k],y)
                μ = lda[k].pmeans
                for (i,l) in enumerate(testlabel) 
                    d = dropdims(sum(abs2,Zp[:,i:i] .- μ,dims=1),dims=1)
                    perf[j,kk,r] += argmin(d) == l
                end
                perf[j,kk,r] /= _ntest 
            end
        end
    end
    TimeResolvedDecoder(fa, testbins, lda, perf)
end

end # module TimeResolvedDecoding
