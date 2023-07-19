"""
    RNNTools

A julia module for training discrete-time and continuous-time RNNs and neural ODEs.

"""

module  RNNTools

using   IterTools: ncycle, NCycle
using   Parameters: @with_kw, @unpack
using   BSON: @save, @load
using   Random: AbstractRNG, shuffle!, seed!, GLOBAL_RNG, randn!
using   Flux: glorot_uniform, glorot_normal, kaiming_uniform, kaiming_normal
using   Flux,
        ForwardDiff,
        OrdinaryDiffEq,
        DiffEqFlux, 
        DiffEqSensitivity, 
        LinearAlgebra,
        Statistics,
        StatsBase,
        Zygote,
        Distributions,
        DataInterpolations,
        ProgressMeter,
        CUDA
CUDA.allowscalar(false)

abstract type Args end

# hyperparameters for data generation
@with_kw mutable struct N_Bit_Task_HPs <: Args
    seed::Int = 0                                                                   # random seed
    ntimebins::Int = 100                                                            # number of time-bins
    obs_dim::Int = 3                                                                # number of observations
    ntrials::Int = 1                                                                # number of trials
    tmin::Float32 = 0.0f0                                                           # time lower bound within a trial
    tmax::Float32 = 1.0f0                                                           # time upper bound within a trial
    width::Int = 1                                                                  # width of each pulse
    frequency::Float64 = 12                                                         # pulses/sec
end

# hyperparameters specific to a single session in training
@with_kw mutable struct Session <: Args
    trange_min::Float32 = 0.0f0                                                     # lower bound of the time range considered for the current training session
    trange_max::Float32 = 1.0f0                                                     # upper bound of the time range considered for the current training session
    max_iter::Int = 100                                                             # maximum number of iterations for the current training session
end

# hyperparameters for training discrete-time and continuous-time RNNs and neural ODEs
@with_kw mutable struct HPs <: Args
    seed::Int = 0                                                                   # random seed
    cuda::Bool = true                                                               # train on gpu if available
    latent_dim::Int = 30                                                            # number of latents assumed by the model
    augmented_dim::Int = 0                                                          # number of augmented dimensions to the latents -- not yet implemented
    init_type::String = "glorot_uniform"                                            # initialization of network state
    τ::Float32 = 0.01f0                                                             # time scale
    tmin::Float32 = 0.0f0                                                           # lower bound of integration
    tmax::Float32 = 1.0f0                                                           # upper bound of integration
    solver::String = "Euler()"                                                      # Euler method; change to Tsit5() to use Tsitouras 5/4 Runge-Kutta method
    sensealg::String = "default"                                                    # default sensitivity algorithm
    dynamics_nn_architecture::Vector{Int} = Int[]                                   # specifications for the FNN parametrizing the dynamics; the length of the vector is the number of hidden layers, with each element being the number of hidden units in each layer; empty set defaults to a linear mapping
    gating_nn_architecture::Vector{Int} = Int[]                                     # specifications for the FNN parametrizing the gating function
    map_nn_architecture::Vector{Int} = Int[]                                        # specifications for the FNN parametrizing the mapping from latents to observations
    encoder_nn_architecture::Vector{Int} = Int[]                                    # specifications for the FNN parametrizing the encoder inferring the initial condition
    training_sessions::Vector{Session} = [Session()]                                # training sessions to execute
    verbose::Bool = true                                                            # show progress meter and loss every iteration
    learning_rate::Float64 = 0.01                                                   # learning rate of ADAMW
    decay_rate::Float64 = 1e-1                                                      # decay rate of ADAMW
    batchsize::Int = 100                                                            # batch size
    dropout_p::Float64 = 0.0                                                        # dropout probability
    clipthresh::Float32 = 1000000.f0                                                # gradient norm threshold for gradient clipping
    layernorm::Bool = false                                                         # apply layer normalization
    hidden_nonlinearity::String = "relu"                                            # activation function used in the hidden layers of the FNN
    gating_nonlinearity::String = "identity"                                        # activation function at the final layer of the FNN parametrizing the gating function; if "identity", we don't apply gating
    dynamics_nonlinearity::String = "identity"                                      # activation function applied at the final layer of the FNN parametrizing the dynamics
    interp::String = "Constant"                                                     # method of data points interpolation
    gru::Bool = false                                                               # specify whether or not the model to be trained is a GRU
    leak::Bool = true                                                               # add a leak term
    gain::Bool = false                                                              # add a gain term
    obs_initialize::Bool = false                                                    # assume that the initial state of the network is an affine-transformation of the observed data at the first time-bin; when false, use input data at the first time-bin
    use_only_last_state::Bool = false                                               # network generates output only at the last time-bin; when false, network generates output at all time-bins
end

include("utils.jl")                                                                 # utility functions
include("generate_data.jl")                                                         # functions to generate data
include("loss_functions.jl")                                                        # defines loss functions
include("mydataloader.jl")                                                          # defines functions for mini-batching data
include("mysolve.jl")                                                               # defines optimization routines
include("model.jl")                                                                 # functions for building network models
include("train.jl")                                                                 # functions for training RNNs and neural ODEs
include("evaluate.jl")                                                              # functions for evaluating RNNs and neural ODEs

export  Session,
        HPs,
        generate_data,
        train,
        evaluate,
        train_rnn,
        evaluate_rnn,
        train_neural_ode,
        evaluate_neural_ode,
        get_dynamics_fun, 
        misssum, 
        missmean, 
        missstd, 
        missstderr,
        mean,
        median,
        compute_R²,
        meshgrid,
        glorot_uniform, 
        glorot_normal, 
        kaiming_uniform, 
        kaiming_normal,
        @save,
        @load

end