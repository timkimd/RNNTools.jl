function loss_rnn(
    p::AbstractVector,
    data::AbstractArray,
    map_hidden_to_obs::DiffEqFlux.FastLayer,
    rnn::DiffEqFlux.FastLayer,
    D::Int,
    dt::AbstractFloat,
    lps::Tuple{Vararg{Int}};
    loss_fun::LossFun = Flux.Losses.mse #Flux.Losses.huber_loss
) where {LossFun}
    N = size(data, 2)
    target, inputs, masks = split_data(data, D)
    rnn.state = reduce(hcat, [rnn.cell.state0 for _ in 1:size(inputs, 2)])
    h = unroll(rnn, inputs, dt, p, lps)
    pred = reshape(
        map_hidden_to_obs(
            reshape(h, size(h, 1), :), 
            p[indx(lps,1)]
        ),
        D,
        N,
        :
    )
    loss = loss_fun(pred[masks], target[masks])
    return loss, h, pred, nothing
end

function loss_rnn(
    p::AbstractVector,
    data::AbstractArray,
    map_hidden_to_obs::DiffEqFlux.FastLayer,
    map_obs_to_hidden::DiffEqFlux.FastLayer,
    obs_initialize::Bool,
    rnn::DiffEqFlux.FastLayer,
    D::Int,
    dt::AbstractFloat,
    lps::Tuple{Vararg{Int}};
    loss_fun::LossFun = Flux.Losses.mse #Flux.Losses.huber_loss
) where {LossFun}
    N = size(data, 2)
    target, inputs, masks = split_data(data, D)
    rnn.state = obs_initialize ? 
        map_obs_to_hidden(target[:,:,1], p[indx(lps,length(lps))]) :
        map_obs_to_hidden(inputs[:,:,1], p[indx(lps,length(lps))])
    h = unroll(rnn, inputs, dt, p, lps)
    pred = reshape(
        map_hidden_to_obs(
            reshape(h, size(h, 1), :), 
            p[indx(lps,1)]
        ),
        D,
        N,
        :
    )
    loss = loss_fun(pred[masks], target[masks])
    return loss, h, pred, nothing
end

function loss_rnn_last(
    p::AbstractVector,
    data::AbstractArray,
    map_hidden_to_obs::DiffEqFlux.FastLayer,
    map_obs_to_hidden::DiffEqFlux.FastLayer,
    obs_initialize::Bool,
    rnn::DiffEqFlux.FastLayer,
    D::Int,
    dt::AbstractFloat,
    lps::Tuple{Vararg{Int}};
    loss_fun::LossFun = Flux.Losses.logitcrossentropy
) where {LossFun}
    N = size(data, 2)
    target, inputs, _ = split_data(data, D)
    rnn.state = obs_initialize ? 
        map_obs_to_hidden(target[:,:,1], p[indx(lps,length(lps))]) :
        map_obs_to_hidden(inputs[:,:,1], p[indx(lps,length(lps))])
    h = unroll(rnn, inputs, dt, p, lps)
    pred = map_hidden_to_obs(
        h[:,:,end], 
        p[indx(lps,1)]
    )
    loss = loss_fun(pred, target[:,:,end] .> 0)
    return loss, h, pred, nothing
end

function loss_rnn_last(
    p::AbstractVector,
    data::AbstractArray,
    map_hidden_to_obs::DiffEqFlux.FastLayer,
    rnn::DiffEqFlux.FastLayer,
    D::Int,
    dt::AbstractFloat,
    lps::Tuple{Vararg{Int}};
    loss_fun::LossFun = Flux.Losses.logitcrossentropy
) where {LossFun}
    N = size(data, 2)
    target, inputs, _ = split_data(data, D)
    rnn.state = reduce(hcat, [rnn.cell.state0 for _ in 1:size(inputs, 2)])
    h = unroll(rnn, inputs, dt, p, lps)
    pred = map_hidden_to_obs(
        h[:,:,end], 
        p[indx(lps,1)]
    )
    loss = loss_fun(pred, target[:,:,end] .> 0)
    return loss, h, pred, nothing
end

function loss_neural_ode(
    p::AbstractVector,
    data::AbstractArray,
    map_latent_to_obs::DiffEqFlux.FastLayer,
    init_h::AbstractVector,
    lps::Tuple{Vararg{Int}},
    tmin::AbstractFloat, 
    tmax::AbstractFloat,
    tsteps::AbstractVector, 
    D::Int,
    F::Function,
    prob::ODEProblem,
    solver::DiffEqBase.AbstractODEAlgorithm, 
    sensealg::SenseAlgFun,
    interp_fun::InterpFun;
    loss_fun::LossFun = Flux.Losses.mse
) where {SenseAlgFun, InterpFun, LossFun}
    N = size(data, 2)
    x, inputs, masks = split_data(data, D) 
    u0s_ = reduce(hcat, [init_h for _ in 1:size(x, 2)])

    if isnothing(inputs)
        z_hat = solve(
            remake(
                prob;
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    else
        # do interpolation on inputs
        interp = isa(interp_fun, typeof(ConstantInterpolation)) ? 
            t -> myConstantInterpolation(inputs, tsteps[tmin .≤ tsteps .≤ tmax], t) :
            interp_fun(
                [inputs[:,:,t] for t = 1:size(inputs, 3)], 
                tsteps[tmin .≤ tsteps .≤ tmax]
            )

        z_hat = solve(
            remake(
                prob;
                f = ((u, p, t) -> F(u, p, t, interp)),
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    end    

    if any(z_hat.retcode .!= :Success)
        return Inf
    end

    nfe = z_hat.destats.nf
    logλ_hat = reshape(
        map_latent_to_obs(reduce(hcat, z_hat.u), p[indx(lps, 2)]), 
        D, 
        N, 
        :
    )

    loss = loss_fun(logλ_hat[masks], x[masks])
    return loss, z_hat, logλ_hat, nfe
end

function loss_neural_ode(
    p::AbstractVector,
    data::AbstractArray,
    map_latent_to_obs::DiffEqFlux.FastLayer,
    map_obs_to_latent::DiffEqFlux.FastLayer,
    obs_initialize::Bool,
    lps::Tuple{Vararg{Int}},
    tmin::AbstractFloat, 
    tmax::AbstractFloat,
    tsteps::AbstractVector, 
    D::Int,
    F::Function,
    prob::ODEProblem,
    solver::DiffEqBase.AbstractODEAlgorithm, 
    sensealg::SenseAlgFun,
    interp_fun::InterpFun;
    loss_fun::LossFun = Flux.Losses.mse
) where {SenseAlgFun, InterpFun, LossFun}
    N = size(data, 2)
    x, inputs, masks = split_data(data, D) 
    u0s_ = obs_initialize ? 
        map_obs_to_latent(x[:,:,1], p[indx(lps, 3)]) :
        map_obs_to_latent(inputs[:,:,1], p[indx(lps, 3)])

    if isnothing(inputs)
        z_hat = solve(
            remake(
                prob;
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    else
        # do interpolation on inputs
        interp = isa(interp_fun, typeof(ConstantInterpolation)) ? 
            t -> myConstantInterpolation(inputs, tsteps[tmin .≤ tsteps .≤ tmax], t) :
            interp_fun(
                [inputs[:,:,t] for t = 1:size(inputs, 3)], 
                tsteps[tmin .≤ tsteps .≤ tmax]
            )

        z_hat = solve(
            remake(
                prob;
                f = ((u, p, t) -> F(u, p, t, interp)),
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    end    

    if any(z_hat.retcode .!= :Success)
        return Inf
    end

    nfe = z_hat.destats.nf
    logλ_hat = reshape(
        map_latent_to_obs(reduce(hcat, z_hat.u), p[indx(lps, 2)]), 
        D, 
        N, 
        :
    )

    loss = loss_fun(logλ_hat[masks], x[masks])
    return loss, z_hat, logλ_hat, nfe
end

function loss_neural_ode_last(
    p::AbstractVector,
    data::AbstractArray,
    map_latent_to_obs::DiffEqFlux.FastLayer,
    map_obs_to_latent::DiffEqFlux.FastLayer,
    obs_initialize::Bool,
    lps::Tuple{Vararg{Int}},
    tmin::AbstractFloat, 
    tmax::AbstractFloat,
    tsteps::AbstractVector, 
    D::Int,
    F::Function,
    prob::ODEProblem,
    solver::DiffEqBase.AbstractODEAlgorithm, 
    sensealg::SenseAlgFun,
    interp_fun::InterpFun;
    loss_fun::LossFun = Flux.Losses.logitcrossentropy
) where {SenseAlgFun, InterpFun, LossFun}
    N = size(data, 2)
    x, inputs, masks = split_data(data, D) 
    u0s_ = obs_initialize ? 
        map_obs_to_latent(x[:,:,1], p[indx(lps, 3)]) :
        map_obs_to_latent(inputs[:,:,1], p[indx(lps, 3)])

    if isnothing(inputs)
        z_hat = solve(
            remake(
                prob;
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    else
        # do interpolation on inputs
        interp = isa(interp_fun, typeof(ConstantInterpolation)) ? 
            t -> myConstantInterpolation(inputs, tsteps[tmin .≤ tsteps .≤ tmax], t) :
            interp_fun(
                [inputs[:,:,t] for t = 1:size(inputs, 3)], 
                tsteps[tmin .≤ tsteps .≤ tmax]
            )

        z_hat = solve(
            remake(
                prob;
                f = ((u, p, t) -> F(u, p, t, interp)),
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    end    

    if any(z_hat.retcode .!= :Success)
        return Inf
    end

    nfe = z_hat.destats.nf
    logλ_hat = map_latent_to_obs(z_hat.u[end], p[indx(lps, 2)])

    loss = loss_fun(logλ_hat, x[:,:,end])
    return loss, z_hat, logλ_hat, nfe
end

function loss_neural_ode_last(
    p::AbstractVector,
    data::AbstractArray,
    map_latent_to_obs::DiffEqFlux.FastLayer,
    init_h::AbstractVector,
    lps::Tuple{Vararg{Int}},
    tmin::AbstractFloat, 
    tmax::AbstractFloat,
    tsteps::AbstractVector, 
    D::Int,
    F::Function,
    prob::ODEProblem,
    solver::DiffEqBase.AbstractODEAlgorithm, 
    sensealg::SenseAlgFun,
    interp_fun::InterpFun;
    loss_fun::LossFun = Flux.Losses.logitcrossentropy
) where {SenseAlgFun, InterpFun, LossFun}
    N = size(data, 2)
    x, inputs, masks = split_data(data, D) 
    u0s_ = reduce(hcat, [init_h for _ in 1:size(x, 2)])

    if isnothing(inputs)
        z_hat = solve(
            remake(
                prob;
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    else
        # do interpolation on inputs
        interp = isa(interp_fun, typeof(ConstantInterpolation)) ? 
            t -> myConstantInterpolation(inputs, tsteps[tmin .≤ tsteps .≤ tmax], t) :
            interp_fun(
                [inputs[:,:,t] for t = 1:size(inputs, 3)], 
                tsteps[tmin .≤ tsteps .≤ tmax]
            )

        z_hat = solve(
            remake(
                prob;
                f = ((u, p, t) -> F(u, p, t, interp)),
                u0=u0s_, 
                tspan=(tsteps[1], tmax), 
                p=p
            ), 
            solver,
            saveat=tsteps[tmin .≤ tsteps .≤ tmax],
            sensealg=sensealg,
            reltol=1e-6, 
            abstol=1e-6
        )
    end    

    if any(z_hat.retcode .!= :Success)
        return Inf
    end

    nfe = z_hat.destats.nf
    logλ_hat = map_latent_to_obs(z_hat.u[end], p[indx(lps, 2)])

    loss = loss_fun(logλ_hat, x[:,:,end])
    return loss, z_hat, logλ_hat, nfe
end