using RNNTools
using Random

idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"]) # 1--27

Random.seed!(8)

cuda = true
layernorm = false
task = RNNTools.N_Bit_Task_HPs(
    seed= 0,
    obs_dim = 2,
    ntimebins = 100,
    ntrials = 600,
    tmin = 0.0f0,
    tmax = 1.0f0,
    width = 2,
    frequency = 12
)

data, target = RNNTools.generate_data_n_bit_task_varying_amp(task)

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

learning_rates = exp10.([range(-4,-2, length=3);])
decay_rates = exp10.([range(-3,-1, length=3);])
batch_sizes = [10; 50; 100]

hps = []
for i = 1:3
    for j = 1:3
        for k = 1:3
            push!(hps, (learning_rates[i], decay_rates[j], batch_sizes[k]))
        end
    end
end

hp = hps[idx]
learning_rate = hp[1]
decay_rate = hp[2]
batch_size = hp[3]

# Gated Neural ODE
args = HPs(
    τ= 0.01f0,
    seed = 0,
    cuda = cuda,
    tmin = 0.f0,
    tmax = 1.0f0,
    latent_dim = 2,
    training_sessions = sessions,
    solver = "Euler()",
    dynamics_nn_architecture = [316, 316],
    gating_nonlinearity = "sigmoid",
    hidden_nonlinearity = "relu",
    learning_rate = learning_rate,
    decay_rate = decay_rate,
    batchsize = batch_size,
    layernorm = layernorm
)

θ_opt_gnode, losses_gnode, val_losses_gnode, lps_gnode, grad_norms_gnode, time_gnode, θ_init_gnode = train(
    args; 
    inputs=data_train, 
    observations=target_train, 
    val_inputs=data_test, 
    val_observations=target_test
)

@save "two_bit_task_varying_amp_data.bson" task data target
@save "two_stats_varying_amp_"*string(idx)*".bson" losses_gnode val_losses_gnode grad_norms_gnode time_gnode lps_gnode
@save "two_params_varying_amp_"*string(idx)*".bson" θ_opt_gnode θ_init_gnode