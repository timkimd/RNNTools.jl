# Adapted from Flux.Data.DataLoader
struct MyDataLoader{D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    rng::R
end

"""
    MyDataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
An object that iterates over mini-batches of `data`, 
each mini-batch containing `batchsize` observations
(except possibly the last one).
Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
The second to last dimension in each tensor is the observation dimension, i.e. the one
divided into mini-batches.
If `shuffle=true`, it shuffles the observations each time iterations are re-started.
If `partial=false` and the number of observations is not divisible by the batchsize, 
then the last mini-batch is dropped.
The original data is preserved in the `data` field of MyDataLoader.
"""
function MyDataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
    ndims(data) > 1 || throw(ArgumentError("Data needs to be more than 1 dimensional"))

    n = _nobs(data)
    if n < batchsize
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    MyDataLoader(data, batchsize, n, partial, imax, [1:n;], shuffle, rng)
end

Base.@propagate_inbounds function Base.iterate(d::MyDataLoader, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    batch = _getobs(d.data, ids)
    return (batch, nexti)
end

function Base.length(d::MyDataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

_nobs(data::AbstractArray) = size(data)[end-1]
_getobs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 2))..., i, :]

Base.eltype(::MyDataLoader{D}) where D = D