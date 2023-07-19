using RNNTools

cuda = true # use GPU if GPU is available

N = 200 # consider only the first 100 trials for training
target = randn(2, 1000, 100) # [observation dimension x number of trials x number of time-bins]
inputs = randn(4, 1000, 100) # [input dimension x number of trials x number of time-bins]

# training dataset
target_train = target[:, 1:N, :]
inputs_train = inputs[:, 1:N, :]

# validation dataset
target_val = target[:, N+1:end, :]
inputs_val = inputs[:, N+1:end, :]

# train from 0s to 1s for 30 epochs
session = Session(
    trange_min = 0.0f0,
    trange_max = 1.0f0,
    max_iter = 30
)

# mutable structure specifying the hyperparameters of our network
args = HPs(
    τ = 0.01f0,                                     # model time constant
    seed = 0,                                       # random seed
    cuda = cuda,                                    # train on GPU if GPU is available
    tmin = 0.f0,                                    # trial start (s)
    tmax = 1.0f0,                                   # trial end (s)
    init_type = "glorot_normal",                    # initialization scheme; Glorot Normal initialization
    latent_dim = 3,                                 # phase-space dimension
    training_sessions = [session],                  # training sessions to be executed
    solver = "Euler()",                             # ODE solver used in training
    gru = false,                                    # is this model a GRU?
    dynamics_nn_architecture = [100, 100],          # dynamics FNN F_θ() architecture; 2 hidden layers with 100 units each layer
    dynamics_nonlinearity = "identity",             # final nonlinearity applied to F_θ() 
    hidden_nonlinearity = "relu",                   # activation function used in between the hidden layers of F_θ()
    gating_nonlinearity = "sigmoid",                # activation function at the final layer of of G_ϕ(); if "identity", we don't apply gating    
    learning_rate = 0.001,                          # learning rate of ADAMW
    decay_rate = 0.1,                               # decay rate of ADAMW
    batchsize = 50                                  # number of trials in a single batch
)

# in the mutable structure above,
# if `gating_nonlinearity = "identity"`, we would have been training a nODE
# if `dynamics_nn_architecture = []`, we would have been training a mGRU
# if `gru = true`, we would have been training a GRU
# if `gating_nonlinearity = "identity"` and `dynamics_nn_architecture = []`, we would have been training a vanilla RNN

h₀ = glorot_normal(args.latent_dim) # initial state of gnODE

# `θ_opt_gnode`     : trained parameters of gnODE
# `losses_gnode`    : training loss traces
# `val_losses_gnode`: validation loss traces
θ_opt_gnode, losses_gnode, val_losses_gnode, _, _, _, _ = train(
    args; 
    init_h=h₀, # if this argument is not given, default to training also the network initial state
    inputs=inputs_train, 
    observations=target_train, 
    val_inputs=inputs_val, 
    val_observations=target_val
)

# `h`   : gnODE state trajectories
# `pred`: output of gnODE
h, pred = evaluate(
    args; 
    init_h=h₀, 
    params=θ_opt_gnode, 
    inputs=inputs_val, 
    observations=target_val
)