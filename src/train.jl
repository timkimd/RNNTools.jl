struct UnstableLossError <: Exception end
struct NotDefinedError <: Exception end

function train(
    args;
    saveat=nothing, 
    inputs=nothing, 
    observations,
    init_params=nothing, 
    init_h=nothing,
    masks=nothing,
    val_masks=nothing,
    val_inputs=nothing,
    val_observations=nothing
)
    if args.solver == "Euler()"
        return train_rnn(
            args; 
            inputs=inputs, 
            observations=observations,
            init_params=init_params,
            init_h=init_h,
            masks=masks,
            val_masks=val_masks,
            val_inputs=val_inputs,
            val_observations=val_observations
        )
    else
        return train_neural_ode(
            args;
            saveat=saveat, 
            inputs=inputs, 
            observations=observations, 
            init_h=init_h,
            init_params=init_params, 
            masks=masks,
            val_masks=val_masks,
            val_inputs=val_inputs,
            val_observations=val_observations
        )
    end
end

function train_rnn(
    args; 
    inputs=nothing, 
    observations,
    init_params=nothing, 
    init_h=nothing,
    masks=nothing,
    val_masks=nothing,
    val_inputs=nothing,
    val_observations=nothing
)
    # load hyperparameter
    args.seed > 0 && seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        args.verbose && @info "Training on GPU"
    else
        device = cpu
        args.verbose && @info "Training on CPU"
    end

    obs_dim = size(observations, 1)
    input_dim = isnothing(inputs) ? 0 : size(inputs, 1)
    this_hidden_nonlinearity = convert_activation_types(args.hidden_nonlinearity)

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
        (glorot_normal(args.latent_dim)  |> device) :
        (init_h |> device)

    # build the map from latent to observation
    ϕ = build_nn(
        args.map_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim,
        obs_dim;
        initW=convert_init_types(args.init_type, args.dynamics_nn_architecture),
        initb=bias_init
    )

    # build the map from observation to latent
    ψ = args.obs_initialize ? 
        build_nn(
            args.encoder_nn_architecture,
            this_hidden_nonlinearity,
            obs_dim,
            args.latent_dim;
            initW=convert_init_types(args.init_type, args.dynamics_nn_architecture),
            initb=bias_init
        ) :
        build_nn(
            args.encoder_nn_architecture,
            this_hidden_nonlinearity,
            input_dim,
            args.latent_dim;
            initW=convert_init_types(args.init_type, args.dynamics_nn_architecture),
            initb=bias_init
        )

    rnn, nns_ = build_rnn(args, input_dim, h₀)
    nns = isnothing(init_h) ?
        (ϕ, nns_..., ψ) :
        (ϕ, nns_...)
    ps = get_initial_params(nns)
    lps = cumsum(map(length, ps))
    if args.init_type == "our_normal"
        θ = reduce(vcat, ps)
        α = Δt/rnn.cell.τ
        if args.gating_nonlinearity == "identity"
            F_theta = (h, x, p) -> (1/α) .* (rnn.cell(h, x, Δt, p, lps) .- h)
        else
            if args.gru
                Wz_ = (h, p) -> rnn.cell.Wz(h, p[indx(lps,2)])
                Uz_ = (x, p) -> rnn.cell.Uz(x, p[indx(lps,5)])

                update_gate = (h, x, p) -> rnn.cell.gating_nonlinearity.(Wz_(h, p) .+ Uz_(x, p))
            else
                Wz_ = (h, p) -> rnn.cell.Wz(h, p[indx(lps,6)])
                Uz_ = (x, p) -> rnn.cell.Uz(x, p[indx(lps,5)])
                g_ = (a, p) -> rnn.cell.g(a, p[indx(lps,7)])

                # not implemented for layernorm
                update_gate = (h, x, p) -> isnothing(rnn.cell.g) ?
                    rnn.cell.gating_nonlinearity.(Wz_(h, p) .+ Uz_(x, p)) :
                    g_(rnn.cell.hidden_nonlinearity.(Wz_(h, p) .+ Uz_(x, p)), p)
            end
            F_theta = (h, x, p) -> (1/α) .* (1 ./ update_gate(h, x, p)) .* (rnn.cell(h, x, Δt, p, lps) .- h)
        end
        if isnothing(inputs)
            J = ForwardDiff.jacobian(h -> F_theta(h, zeros(input_dim), θ), h₀) # may not work, we need to have inputs
        else
            if isnothing(init_h)
                h0 = args.obs_initialize ? 
                    ψ(observations[:,:,1], θ[indx(lps,length(lps))]) :
                    ψ(inputs[:,:,1], θ[indx(lps,length(lps))])
                h0 = mean(h0, dims=2)[:]
            else
                h0 = h₀
            end
            J = ForwardDiff.jacobian(h -> F_theta(h, mean(reshape(inputs, size(inputs,1),:), dims=2)[:], θ), h0)
        end
        η_max = maximum(real.(eigen(J).values))
        if isempty(args.dynamics_nn_architecture)
            L = 1
        else
            L = length(args.dynamics_nn_architecture) + 1
        end
        σ_w = (1.f0/(η_max+1.f0))^(1.f0/L)
        if isnothing(init_h)
            θ[lps[1]:lps[end-1]] .= σ_w .* θ[lps[1]:lps[end-1]]
        else
            θ[lps[1]:lps[end]] .= σ_w .* θ[lps[1]:lps[end]]
        end
        θ = θ |> device
    else
        θ = reduce(vcat, ps) |> device
    end
    # Progress meter
    total_iter = 0
    max_iters = []
    temp = MyDataLoader(observations, batchsize = args.batchsize)
    for (i, session) in enumerate(args.training_sessions)
        @unpack trange_min, trange_max, max_iter = session
        total_iter += length(ncycle(temp, max_iter))
        append!(max_iters, total_iter)
    end

    if args.verbose
        prog = ProgressMeter.Progress(
            total_iter, 
            dt=0.5, 
            barglyphs=BarGlyphs("[=> ]"), 
            barlen=50, 
            color=:yellow; 
            showspeed=true
        )
    end

    # Define optimizer
    θ_min = isnothing(init_params) ? copy(θ) : (init_params |> device)
    opt = Flux.Optimise.Optimiser(
        ADAMW(args.learning_rate, (0.9, 0.999), args.decay_rate), 
        ClipNorm(args.clipthresh)
    )
    observations_ = concat_data(observations, inputs, masks)
    val_observations_ = isnothing(val_observations) ?
        nothing : 
        concat_data(val_observations, val_inputs, val_masks) |> device

    # Callback function
    losses = []
    val_losses = []
    grad_norms = []
    val_err = typemax(eltype(θ))
    val_θ = cpu(θ)
    cb = function (p, g, l, no)
        append!(losses, l)
        append!(grad_norms, norm(g))
        args.verbose && next!(prog; showvalues = [(:Loss, l)])
        if !isnothing(val_observations)
            if args.use_only_last_state
                val_loss, _, _ = isnothing(init_h) ? 
                    loss_rnn_last(
                        p, 
                        val_observations_,
                        ϕ,
                        ψ, 
                        args.obs_initialize,
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ) :   
                    loss_rnn_last(
                        p, 
                        val_observations_,
                        ϕ, 
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    )
            else
                val_loss, _, _ = isnothing(init_h) ? 
                    loss_rnn(
                        p, 
                        val_observations_,
                        ϕ,
                        ψ, 
                        args.obs_initialize,
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ) :   
                    loss_rnn(
                        p, 
                        val_observations_,
                        ϕ, 
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    )
            end
            append!(val_losses, val_loss)
            if val_loss < val_err  #found a better solution
                val_err = val_loss
                val_θ = cpu(p)
            end
        end
        
        return false
    end

    stime = time()
    for (i, session) in enumerate(args.training_sessions)
        @unpack trange_min, trange_max, max_iter = session

        # train only from trange_min to trange_max
        observations_session = observations_[:, :, trange_min .≤ saveat .≤ trange_max]

        # find the right model for the input and observation types given
        train_loader = MyDataLoader(observations_session, batchsize = args.batchsize)
        if args.use_only_last_state
            θ_min = isnothing(init_h) ? 
                mysolve(
                    (x, batch) -> loss_rnn_last(
                        x, 
                        batch,
                        ϕ,
                        ψ, 
                        args.obs_initialize,
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                ) :   
                mysolve(
                    (x, batch) -> loss_rnn_last(
                        x, 
                        batch,
                        ϕ, 
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                )
        else
            θ_min = isnothing(init_h) ? 
                mysolve(
                    (x, batch) -> loss_rnn(
                        x, 
                        batch,
                        ϕ,
                        ψ, 
                        args.obs_initialize,
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                ) :   
                mysolve(
                    (x, batch) -> loss_rnn(
                        x, 
                        batch,
                        ϕ, 
                        rnn,
                        obs_dim,
                        Δt,
                        lps
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                )
        end
    end
    etime = time() - stime
    θ_opt = isnothing(val_observations) ? cpu(θ_min) : val_θ
    return θ_opt, losses, val_losses, lps, grad_norms, etime, cpu(θ)
end

function train_neural_ode(
    args;
    saveat=nothing, 
    inputs=nothing, 
    observations, 
    init_h=nothing,
    init_params=nothing, 
    masks=nothing,
    val_masks=nothing,
    val_inputs=nothing,
    val_observations=nothing
)
    # load hyperparameter
    args.seed > 0 && seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        args.verbose && @info "Training on GPU"
    else
        device = cpu
        args.verbose && @info "Training on CPU"
    end

    obs_dim = size(observations, 1)
    input_dim = isnothing(inputs) ? 0 : size(inputs, 1)
    this_solver = convert_solver_types(args.solver)
    this_sensealg = convert_sensealg_types(args.sensealg)
    this_hidden_nonlinearity = convert_activation_types(args.hidden_nonlinearity)
    this_gating_nonlinearity = convert_activation_types(args.gating_nonlinearity)
    this_dynamics_nonlinearity = convert_activation_types(args.dynamics_nonlinearity)
    this_interp = convert_interp_types(args.interp)   

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

    # initialize states
    u₀ = isnothing(init_h) ? 
        (glorot_normal(args.latent_dim)  |> device) :
        (init_h |> device)

    _, nns = build_rnn(args, input_dim, u₀)
    ps = get_initial_params(nns)
    lps = cumsum(map(length, ps))
    θ = reduce(vcat, ps)
    θ_ƒ = θ[1:lps[3]]
    θ_𝘨 = isa(this_gating_nonlinearity, typeof(identity)) ? 
        [] :
        θ[lps[3]+1:lps[6]]

    # build dynamics function
    ƒ = build_nn(
        args.dynamics_nn_architecture,
        this_hidden_nonlinearity,
        args.latent_dim + input_dim,
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
        obs_dim;
        initW=convert_init_types(args.init_type, args.dynamics_nn_architecture),
        initb=bias_init
    )

    # build the map from observation to latent
    x_dim = args.obs_initialize ? obs_dim : input_dim
    ψ = build_nn(
        args.encoder_nn_architecture,
        this_hidden_nonlinearity,
        x_dim,
        args.latent_dim;
        initW=convert_init_types(args.init_type, args.dynamics_nn_architecture),
        initb=bias_init
    )

    if isnothing(init_h)
        # store the functions as a tuple
        nns = (ƒ, ϕ, ψ, 𝘨)

        # get the functions' initial parameters and concatenate the trainable parameters
        ps = get_initial_params(nns)
        lps = cumsum(map(length, ps))
        ps[1] .= θ_ƒ
        if !isa(this_gating_nonlinearity, typeof(identity))
            ps[4] .= θ_𝘨
        end
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
                        (-1/args.τ) .* ( u .- p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,4)]) .* 
                        (-1 .* ( u .- p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,4)]) .* 
                        ( p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) )
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
                        (-1/args.τ) .* ( u .- ƒ(vcat(A(t),u), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,4)]) .* 
                        (-1 .* ( u .- ƒ(vcat(A(t),u), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* ƒ(vcat(A(t),u), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,4)]) .* 
                        ( ƒ(vcat(A(t),u), p[indx(lps,1)]) )
            end
        end

        # parameters of the model
        if args.gain
            θ = isa(this_gating_nonlinearity, typeof(identity)) ? 
                (Float32[reduce(vcat, ps[1:3]);1.f0] |> device) :
                (Float32[reduce(vcat, ps);1.f0] |> device)
        else 
            θ = isa(this_gating_nonlinearity, typeof(identity)) ? 
                (reduce(vcat, ps[1:3]) |> device) :
                (reduce(vcat, ps) |> device)
        end
    else
        # store the functions as a tuple
        nns = (ƒ, ϕ, 𝘨)

        # get the functions' initial parameters and concatenate the trainable parameters
        ps = get_initial_params(nns)
        lps = cumsum(map(length, ps))
        ps[1] .= θ_ƒ
        if !isa(this_gating_nonlinearity, typeof(identity))
            ps[3] .= θ_𝘨
        end

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
                        (-1/args.τ) .* ( u .- p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,3)]) .* 
                        (-1 .* ( u .- p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,3)]) .* 
                        ( p[end:end] .* ƒ(vcat(A(t),u), p[indx(lps,1)]) )
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
                        (-1/args.τ) .* ( u .- ƒ(vcat(A(t),u), p[indx(lps,1)]) ) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,3)]) .* 
                        (-1 .* ( u .- ƒ(vcat(A(t),u), p[indx(lps,1)]) ) )
            else
                F_w_input = isa(this_gating_nonlinearity, typeof(identity)) ? 
                    (u, p, t, A) -> 
                        (1/args.τ) .* ƒ(vcat(A(t),u), p[indx(lps,1)]) :
                    (u, p, t, A) -> 
                        (1/args.τ) .* 𝘨(vcat(A(t),u), p[indx(lps,3)]) .* 
                        ( ƒ(vcat(u, A(t)), p[indx(lps,1)]) )
            end
        end

        # parameters of the model
        if args.gain
            θ = isa(this_gating_nonlinearity, typeof(identity)) ? 
                (Float32[reduce(vcat, ps[1:2]);1.f0] |> device) :
                (Float32[reduce(vcat, ps);1.f0] |> device)
        else 
            θ = isa(this_gating_nonlinearity, typeof(identity)) ? 
                (reduce(vcat, ps[1:2]) |> device) :
                (reduce(vcat, ps) |> device)
        end
    end

    # do interpolation on inputs
    interp = isa(this_interp, typeof(ConstantInterpolation)) ? 
        t -> myConstantInterpolation(inputs, saveat, t) :
        this_interp([inputs[:,:,t] for t = 1:size(inputs, 3)], saveat)

    # find the right model for the input and observation types given
    prob_node = isnothing(inputs) ? 
        ODEProblem(F, u₀, tspan, θ) : 
        ODEProblem((u, p, t) -> F_w_input(u, p, t, interp), u₀, tspan, θ)

    opt = Flux.Optimise.Optimiser(
        ADAMW(args.learning_rate, (0.9, 0.999), args.decay_rate), 
        ClipNorm(args.clipthresh)
    )
    observations_ = concat_data(observations, inputs, masks)
    val_observations_ = isnothing(val_observations) ? 
        nothing : 
        concat_data(val_observations, val_inputs, val_masks) |> device
    train_loader_full = MyDataLoader(observations_, batchsize = args.batchsize)
    θ_min = isnothing(init_params) ? copy(θ) : (init_params |> device)

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

    # Progress meter
    total_iter = 0
    max_iters = [0]
    for (i, session) in enumerate(args.training_sessions)
        @unpack trange_min, trange_max, max_iter = session
        total_iter += length(ncycle(train_loader_full, max_iter))
        append!(max_iters, total_iter)
    end

    if args.verbose
        prog = ProgressMeter.Progress(
            total_iter, 
            dt=0.5, 
            barglyphs=BarGlyphs("[=> ]"), 
            barlen=50, 
            color=:yellow; 
            showspeed=true
        )
    end

    # Callback function
    losses = []
    nfes = []
    val_losses = []
    val_nfes = []
    grad_norms = []
    val_err = typemax(eltype(θ))
    val_θ = cpu(θ)
    cb = function (p, g, l, nfe)
        cut = length(losses) > 3 ? max_iters[sum(length(losses) .> max_iters)] : 1
        if (length(losses[cut+1:end]) > 5) && (isnan(l) || isinf(l)) 
            args.verbose && @info "Current session aborted..." #" Restarting session with decreased learning rate..."    
            losses = losses[1:cut]
            args.verbose && update!(prog, cut)
            throw(UnstableLossError())
        else
            append!(losses, l)
            append!(nfes, nfe)
            append!(grad_norms, norm(g))
            args.verbose && next!(prog; showvalues = [(:Loss, l)])
            if !isnothing(val_observations)
                if args.use_only_last_state
                    val_loss, _, _, val_nfe = isnothing(init_h) ?
                        loss_neural_ode_last(
                            p, 
                            val_observations_, 
                            nns[2:3]...,
                            args.obs_initialize,
                            lps,
                            args.tmin,
                            args.tmax,
                            common_loss_args...
                        ) :
                        loss_neural_ode_last(
                        p, 
                        val_observations_, 
                        nns[2],
                        u₀,
                        lps,
                        args.tmin,
                        args.tmax,
                        common_loss_args...
                    )
                else
                    val_loss, _, _, val_nfe = isnothing(init_h) ?
                        loss_neural_ode(
                            p, 
                            val_observations_, 
                            nns[2:3]...,
                            args.obs_initialize,
                            lps,
                            args.tmin,
                            args.tmax,
                            common_loss_args...
                        ) :
                        loss_neural_ode(
                            p, 
                            val_observations_, 
                            nns[2],
                            u₀,
                            lps,
                            args.tmin,
                            args.tmax,
                            common_loss_args...
                        )
                end
                append!(val_losses, val_loss)
                append!(val_nfes, val_nfe)
                if val_loss < val_err  #found a better solution
                    val_err = val_loss
                    val_θ = cpu(p)
                end
            end
        end
        return false
    end

    # training loop
    stime = time()
    for (i, session) in enumerate(args.training_sessions)
        @unpack trange_min, trange_max, max_iter = session
        observations_session = observations_[:, :, trange_min .≤ saveat .≤ trange_max]
        train_loader = MyDataLoader(observations_session, batchsize = args.batchsize)

        if args.use_only_last_state
            θ_min = isnothing(init_h) ? 
                mysolve(
                    (x, batch) -> loss_neural_ode_last(
                        x, 
                        batch, 
                        nns[2:3]...,
                        args.obs_initialize,
                        lps, 
                        trange_min, 
                        trange_max,
                        common_loss_args...
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                ) :
                mysolve(
                (x, batch) -> loss_neural_ode_last(
                    x, 
                    batch, 
                    nns[2],
                    u₀,
                    lps, 
                    trange_min, 
                    trange_max,
                    common_loss_args...
                ),
                θ_min,
                opt, 
                ncycle(train_loader, max_iter),
                device,
                cb = cb
            )
        else
            θ_min = isnothing(init_h) ? 
                mysolve(
                    (x, batch) -> loss_neural_ode(
                        x, 
                        batch, 
                        nns[2:3]...,
                        args.obs_initialize,
                        lps, 
                        trange_min, 
                        trange_max,
                        common_loss_args...
                    ),
                    θ_min,
                    opt, 
                    ncycle(train_loader, max_iter),
                    device,
                    cb = cb
                ) :
                mysolve(
                (x, batch) -> loss_neural_ode(
                    x, 
                    batch, 
                    nns[2],
                    u₀,
                    lps, 
                    trange_min, 
                    trange_max,
                    common_loss_args...
                ),
                θ_min,
                opt, 
                ncycle(train_loader, max_iter),
                device,
                cb = cb
            )
        end
    end
    etime = time() - stime
    θ_opt = isnothing(val_observations) ? cpu(θ_min) : val_θ
    return θ_opt, losses, val_losses, lps, grad_norms, etime, cpu(θ), nfes, val_nfes
end