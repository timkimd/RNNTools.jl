function evaluate(
    args; 
    saveat=nothing,
    inputs=nothing, 
    observations,
    params, 
    init_h=nothing
)
    if args.solver == "Euler()"
        return evaluate_rnn(
            args; 
            inputs=inputs, 
            observations=observations,
            params=params, 
            init_h=init_h
        )
    else
        return evaluate_neural_ode(
            args; 
            saveat=saveat,
            inputs=inputs, 
            observations=observations, 
            params=params,
            init_h=init_h
        )
    end
end

function evaluate_rnn(
    args; 
    inputs=nothing, 
    observations,
    params, 
    init_h=nothing
)
    # load hyperparameter
    args.seed > 0 && seed!(args.seed)

    obs_dim = size(observations, 1)
    input_dim = isnothing(inputs) ? 0 : size(inputs, 1)
    this_hidden_nonlinearity = convert_activation_types(args.hidden_nonlinearity)
    this_gating_nonlinearity = convert_activation_types(args.gating_nonlinearity)
    this_dynamics_nonlinearity = convert_activation_types(args.dynamics_nonlinearity)

    tspan = (args.tmin, args.tmax)
    ntimebins = size(observations, 3)
    trange = range(tspan[1], tspan[2], length=ntimebins+1)
    Δt = trange[2] - trange[1]
    saveat = zeros(Float32, ntimebins)
    for i = 1:ntimebins
        saveat[i] = trange[i] + (trange[i+1] - trange[i])/2.f0
    end
    saveat = round.(saveat, digits=Int(abs(floor(log10(Δt))) + 1))

    # initialize states
    h₀ = isnothing(init_h) ? 
        glorot_normal(args.latent_dim) :
        init_h

    # build the map from latent to observation
    ϕ = build_nn(
        args.map_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim,
        obs_dim
    )

    # build the map from observation to latent
    ψ = args.obs_initialize ? 
        build_nn(
            args.encoder_nn_architecture,
            this_hidden_nonlinearity,
            obs_dim,
            args.latent_dim
        ) :
        build_nn(
            args.encoder_nn_architecture,
            this_hidden_nonlinearity,
            input_dim,
            args.latent_dim
        )

    rnn, nns_ = build_rnn(args, input_dim, h₀)
    nns = isnothing(init_h) ?
        (ϕ, nns_..., ψ) :
        (ϕ, nns_...)
    ps = get_initial_params(nns)
    lps = cumsum(map(length, ps))
    
    observations_ = concat_data(observations, inputs)

    if args.use_only_last_state
        if isnothing(init_h) 
            loss, h, pred, _ = loss_rnn_last(
                params, 
                observations_,
                ϕ,
                ψ, 
                args.obs_initialize,
                rnn,
                obs_dim,
                Δt,
                lps
            )
        else 
            loss, h, pred, _ = loss_rnn_last(
                params, 
                observations_,
                ϕ, 
                rnn,
                obs_dim,
                Δt,
                lps
            )
        end
    else
        if isnothing(init_h) 
            loss, h, pred, _ = loss_rnn(
                params, 
                observations_,
                ϕ,
                ψ, 
                args.obs_initialize,
                rnn,
                obs_dim,
                Δt,
                lps
            )
        else  
            loss, h, pred, _ = loss_rnn(
                params, 
                observations_,
                ϕ, 
                rnn,
                obs_dim,
                Δt,
                lps
            )
        end
    end
    
    return h, pred, loss
end

function evaluate_neural_ode(
    args; 
    saveat=nothing,
    inputs=nothing, 
    observations, 
    params,
    init_h=nothing
)
    # load hyperparameter
    args.seed > 0 && seed!(args.seed)

    obs_dim = size(observations, 1)
    input_dim = isnothing(inputs) ? 0 : size(inputs, 1)
    tspan = (args.tmin, args.tmax)
    if isnothing(saveat)
        ntimebins = size(observations, 3)
        trange = range(tspan[1], tspan[2], length=ntimebins+1)
        Δt = trange[2] - trange[1]
        saveat = zeros(Float32, ntimebins)
        for i = 1:ntimebins
            saveat[i] = trange[i] + (trange[i+1] - trange[i])/2.f0
        end
        saveat = round.(saveat, digits=Int(abs(floor(log10(Δt))) + 1))
    end

    this_solver = convert_solver_types(args.solver)
    this_sensealg = convert_sensealg_types(args.sensealg)
    this_hidden_nonlinearity = convert_activation_types(args.hidden_nonlinearity)
    this_gating_nonlinearity = convert_activation_types(args.gating_nonlinearity)
    this_dynamics_nonlinearity = convert_activation_types(args.dynamics_nonlinearity)
    this_interp = convert_interp_types(args.interp)   

    # initialize states
    u₀ = isnothing(init_h) ? 
        glorot_normal(args.latent_dim) :
        init_h

    # build dynamics function
    ƒ = build_nn(
        args.dynamics_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim + input_dim, # this should change
        args.latent_dim;
        layernorm = args.layernorm,
        final_nonlinearity = this_dynamics_nonlinearity
    )

    # build gating function
    𝘨 = build_nn(
        args.gating_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim + input_dim,
        args.latent_dim;
        layernorm = args.layernorm,
        final_nonlinearity = this_gating_nonlinearity
    )

    # build the map from latent to observation
    ϕ = build_nn(
        args.map_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim,
        obs_dim
    )

    # build the map from observation to latent
    x_dim = args.obs_initialize ? obs_dim : input_dim
    ψ = build_nn(
        args.encoder_nn_architecture,
        this_hidden_nonlinearity,
        x_dim,
        args.latent_dim
    )

    if isnothing(init_h)
        # store the functions as a tuple
        nns = (ƒ, ϕ, ψ, 𝘨)

        # get the functions' initial parameters and concatenate the trainable parameters
        ps = get_initial_params(nns)
        lps = cumsum(map(length, ps))

        if args.gain 
            # neural ODE without input (either vanilla or gated)
            if args.leak # also want some gain term in front of f
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (-1/args.τ) .* ( u .- p[end:end] .* ƒ(u,p[indx(lps,1)]) ) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,4)]) .* (-1 .* ( u .- p[end:end] .* ƒ(u,p[indx(lps,1)]) ) )
            else
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (1/args.τ) .* p[end:end] .* ƒ(u,p[indx(lps,1)]) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,4)]) .* ( p[end:end] .* ƒ(u,p[indx(lps,1)]) )
            end

            # neural ODE with inputs (either vanilla or gated)
            if args.leak
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (-1/args.τ) .* ( u .- p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,4)]) .* 
                        (-1 .* ( u .- p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,4)]) .* 
                        ( p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) )
            end
        else
            # neural ODE without input (either vanilla or gated)
            if args.leak # also want some gain term in front of f
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (-1/args.τ) .* ( u .- ƒ(u,p[indx(lps,1)]) ) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,4)]) .* (-1 .* ( u .- ƒ(u,p[indx(lps,1)]) ) )
            else
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (1/args.τ) .* ƒ(u,p[indx(lps,1)]) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,4)]) .* ( ƒ(u,p[indx(lps,1)]) )
            end

            # neural ODE with inputs (either vanilla or gated)
            if args.leak
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (-1/args.τ) .* ( u .- ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,4)]) .* 
                        (-1 .* ( u .- ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,4)]) .* 
                        ( ƒ(vcat(u, A(t)), p[indx(lps,1)]) )
            end
        end
    else
        # store the functions as a tuple
        nns = (ƒ, ϕ, 𝘨)

        # get the functions' initial parameters and concatenate the trainable parameters
        ps = get_initial_params(nns)
        lps = cumsum(map(length, ps))

        if args.gain 
            # neural ODE without input (either vanilla or gated)
            if args.leak # also want some gain term in front of f
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (-1/args.τ) .* ( u .- p[end:end] .* ƒ(u,p[indx(lps,1)]) ) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,3)]) .* (-1 .* ( u .- p[end:end] .* ƒ(u,p[indx(lps,1)]) ) )
            else
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (1/args.τ) .* p[end:end] .* ƒ(u,p[indx(lps,1)]) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,3)]) .* ( p[end:end] .* ƒ(u,p[indx(lps,1)]) )
            end

            # neural ODE with inputs (either vanilla or gated)
            if args.leak
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (-1/args.τ) .* ( u .- p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,3)]) .* 
                        (-1 .* ( u .- p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,3)]) .* 
                        ( p[end:end] .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) )
            end
        else
            # neural ODE without input (either vanilla or gated)
            if args.leak # also want some gain term in front of f
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (-1/args.τ) .* ( u .- ƒ(u,p[indx(lps,1)]) ) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,3)]) .* (-1 .* ( u .- ƒ(u,p[indx(lps,1)]) ) )
            else
                F = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t) -> (1/args.τ) .* ƒ(u,p[indx(lps,1)]) :
                    (u, p, t) -> (1/args.τ) .* 𝘨(u,p[indx(lps,3)]) .* ( ƒ(u,p[indx(lps,1)]) )
            end

            # neural ODE with inputs (either vanilla or gated)
            if args.leak
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (-1/args.τ) .* ( u .- ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,3)]) .* 
                        (-1 .* ( u .- ƒ(vcat(u, A(t)), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* ƒ(vcat(u, A(t)), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(u, A(t)), p[indx(lps,3)]) .* 
                        ( ƒ(vcat(u, A(t)), p[indx(lps,1)]) )
            end
        end
    end

    # do interpolation on inputs
    interp = isa(this_interp, typeof(ConstantInterpolation)) ? 
        t -> myConstantInterpolation(inputs, saveat, t) :
        this_interp([inputs[:,:,t] for t = 1:size(inputs, 3)], saveat)

    # find the right model for the input and observation types given
    prob_node = isnothing(inputs) ? 
        ODEProblem(F, u₀, tspan, params) : 
        ODEProblem((u, p, t) -> F_w_input(u, p, t, interp), u₀, tspan, params)

    observations_ = concat_data(observations, inputs)

    common_loss_args = isnothing(inputs) ?
        (
            saveat, 
            obs_dim,
            F,
            prob_node,
            this_solver, 
            this_sensealg,
            this_interp
        ) :
        (
            saveat, 
            obs_dim,
            F_w_input,
            prob_node,
            this_solver, 
            this_sensealg,
            this_interp
        )

    if args.use_only_last_state
        if isnothing(init_h) 
            loss, z_hat, logλ_hat, nfe = loss_neural_ode_last(
                params, 
                observations_, 
                nns[2:3]...,
                args.obs_initialize,
                lps, 
                args.tmin, 
                args.tmax,
                common_loss_args...
            )
        else
            loss, z_hat, logλ_hat, nfe = loss_neural_ode_last(
                params, 
                observations_, 
                nns[2],
                u₀,
                lps, 
                args.tmin, 
                args.tmax,
                common_loss_args...
            )
        end
    else
        if isnothing(init_h) 
            loss, z_hat, logλ_hat, nfe = loss_neural_ode(
                params, 
                observations_, 
                nns[2:3]...,
                args.obs_initialize,
                lps, 
                args.tmin, 
                args.tmax,
                common_loss_args...
            )
        else
            loss, z_hat, logλ_hat, nfe = loss_neural_ode(
                params, 
                observations_, 
                nns[2],
                u₀,
                lps, 
                args.tmin, 
                args.tmax,
                common_loss_args...
            )
        end
    end

    return Array(z_hat), logλ_hat, loss, nfe
end