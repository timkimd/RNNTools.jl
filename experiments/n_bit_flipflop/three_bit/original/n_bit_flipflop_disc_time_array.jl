using RNNTools
using Random

idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"]) # 1--3

Random.seed!(8)

cuda = true
layernorm = false
task = RNNTools.N_Bit_Task_HPs(
    seed= 0,
    obs_dim = 3,
    ntimebins = 100,
    ntrials = 600,
    tmin = 0.0f0,
    tmax = 1.0f0,
    width = 2,
    frequency = 12
)

data, target = RNNTools.generate_data_n_bit_task(task)

N = 500
data_train = data[:,1:N,:]
target_train = target[:,1:N,:]
data_test = data[:,N+1:end,:]
target_test = target[:,N+1:end,:]

# ********************************************************** Train ******************************************************************** #

session = Session(
    trange_min = 0.0f0,
    trange_max = 1.0f0,
    max_iter = 200
)

sessions = [session]

latent_dims = [6, 12, 18]
latent_dim = latent_dims[idx]

# mGRU
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = latent_dim,
    training_sessions = sessions,
    solver = "Euler()",
    gating_nonlinearity = "sigmoid",
    dynamics_nonlinearity = "tanh",
    learning_rate = 0.01,
    decay_rate = 1e-1,
    batchsize = 100,
    layernorm = layernorm
)

θ_opt_mgru, losses_mgru, val_losses_mgru, lps_mgru, grad_norms_mgru, time_mgru, θ_init_mgru = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

# RNN
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = latent_dim,
    training_sessions = sessions,
    solver = "Euler()",
    dynamics_nonlinearity = "tanh",
    learning_rate = 0.01,
    decay_rate = 1e-1,
    batchsize = 100,
    layernorm = layernorm
)

θ_opt_rnn, losses_rnn, val_losses_rnn, lps_rnn, grad_norms_rnn, time_rnn, θ_init_rnn = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

# GRU
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = latent_dim,
    training_sessions = sessions,
    solver = "Euler()",
    gru=true,
    gating_nonlinearity = "sigmoid",
    dynamics_nonlinearity = "tanh",
    learning_rate = 0.01,
    decay_rate = 1e-1,
    batchsize = 100,
    layernorm = layernorm
)

θ_opt_gru, losses_gru, val_losses_gru, lps_gru, grad_norms_gru, time_gru, θ_init_gru = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

# Neural ODE
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = latent_dim,
    training_sessions = sessions,
    solver = "Euler()",
    dynamics_nn_architecture = [100, 100, 100],
    hidden_nonlinearity = "relu",
    learning_rate = 0.001,
    decay_rate = 1e-1,
    batchsize = 100,
    layernorm = layernorm
)

θ_opt_node, losses_node, val_losses_node, lps_node, grad_norms_node, time_node, θ_init_node = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

# Gated Neural ODE
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = latent_dim,
    training_sessions = sessions,
    solver = "Euler()",
    dynamics_nn_architecture = [100, 100, 100],
    gating_nonlinearity = "sigmoid",
    hidden_nonlinearity = "relu",
    learning_rate = 0.001,
    decay_rate = 1e-1,
    batchsize = 100,
    layernorm = layernorm
)

θ_opt_gnode, losses_gnode, val_losses_gnode, lps_gnode, grad_norms_gnode, time_gnode, θ_init_gnode = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

@save "three_bit_task_data.bson" task data target
@save "rnn_stats_"*string(idx)*".bson" losses_rnn losses_mgru losses_gru losses_node losses_gnode val_losses_rnn val_losses_mgru val_losses_gru val_losses_node val_losses_gnode grad_norms_rnn grad_norms_mgru grad_norms_gru grad_norms_node grad_norms_gnode time_rnn time_mgru time_gru time_node time_gnode lps_rnn lps_mgru lps_gru lps_node lps_gnode
@save "rnn_params_"*string(idx)*".bson" θ_opt_rnn θ_opt_mgru θ_opt_gru θ_opt_node θ_opt_gnode θ_init_rnn θ_init_mgru θ_init_gru θ_init_node θ_init_gnode h0