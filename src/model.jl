const SMALL_VAL = sqrt(eps(Float32))

function σᵦ(x)
    β = 10
    return sigmoid_fast(β*x)
end

function step(x)
    return x > 0 ? 1 : 0
end

step(x::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(step(ForwardDiff.value(x)), ForwardDiff.partials(x))
Zygote.@adjoint step(x)=step(x), y->(y,) # Straight-Through Estimator

function LN(z) # layer normalization following Xu et al., 2019
    μ₀ = mean(z, dims=1)
    σ₀ = std(z, corrected=false, dims=1)
    return (z .- μ₀) ./ (σ₀ .+ SMALL_VAL)
end

mutable struct FastLN{A} <: DiffEqFlux.FastLayer
    activation::A
end

function (f::FastLN)(x, p)
    return f.activation.(LN(x))
end

# adapted from Flux.jl to work with FastChain in DiffEqFlux.jl
mutable struct FastDropout{P} <: DiffEqFlux.FastLayer
    p::P
    active::Bool
end

function FastDropout(p)
    @assert 0 ≤ p ≤ 1
    FastDropout{typeof(p)}(p, true)
end

function (a::FastDropout)(x, p)
    a.active || return x
    y = similar(x)
    rand!(y)
    q = 1 - a.p
    @inbounds for i=1:length(y)
        y[i] = y[i] > a.p ? 1 / q : 0
    end
    return y .* x
end

testmode!(a::FastDropout, test) = (a.active = !test)

mutable struct FastRecur{C,H} <: DiffEqFlux.FastLayer
    cell::C
    state::H
end

function (m::FastRecur)(
    x::AbstractArray, 
    Δt::AbstractFloat, 
    params::AbstractVector, 
    lps::Tuple{Vararg{Int}}
)
    m.state = m.cell(
        m.state, 
        x,
        Δt, 
        params, 
        lps
    )
    return m.state
end

function unroll(
    m::FastRecur,
    x::AbstractArray, 
    Δt::AbstractFloat, 
    params::AbstractVector, 
    lps::Tuple{Vararg{Int}}
)
    h = [m(x_t, Δt, params, lps) for x_t in eachslice(x, dims=3)]
    return reshape(reduce(hcat, h), size(h[1])..., :)
end

# GRU
struct FastGRUCell{A,B,T,H0,F1,F2,Bool}
    Wz::A
    Wr::A
    Wh::A
    Uz::B
    Ur::B
    Uh::B
    τ::T
    state0::H0
    dynamics_nonlinearity::F1
    gating_nonlinearity::F2
    layernorm::Bool
end

function FastGRUCell(
    input_dim::Int, 
    hidden_dim::Int;
    τ=1.f0, 
    state0,
    initW=Flux.glorot_normal, 
    initb=Flux.zeros32,
    dynamics_nonlinearity=tanh_fast,
    gating_nonlinearity=sigmoid_fast,
    layernorm=false
)
    Wz = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)
    Wr = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)
    Wh = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)
    Uz = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)
    Ur = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)
    Uh = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)

    return FastGRUCell(
        Wz, 
        Wr,
        Wh,
        Uz,
        Ur,
        Uh,
        τ,
        state0,
        dynamics_nonlinearity,
        gating_nonlinearity,
        layernorm
    )
end

function (m::FastGRUCell)(
    h::AbstractArray, 
    x::AbstractArray, 
    Δt::AbstractFloat,
    params::AbstractVector,  
    lps::Tuple{Vararg{Int}}
)
    α = Δt/m.τ
    Wzh, Wrh = m.Wz(h, params[indx(lps,2)]), m.Wr(h, params[indx(lps,3)])
    Uzx, Urx = m.Uz(x, params[indx(lps,5)]), m.Ur(x, params[indx(lps,6)])
    z = m.layernorm ? ( m.gating_nonlinearity.(LN(Wzh) .+ Uzx) ) : ( @. m.gating_nonlinearity(Wzh + Uzx) )
    r = m.layernorm ? ( m.gating_nonlinearity.(LN(Wrh) .+ Urx) ) : ( @. m.gating_nonlinearity(Wrh + Urx) )
    Whh, Uhh = m.Wh(h .* r, params[indx(lps,4)]), m.Uh(x, params[indx(lps,7)])
    h̃ = m.layernorm ? ( m.dynamics_nonlinearity.(LN(Whh) .+ Uhh) ) : ( @. m.dynamics_nonlinearity(Whh + Uhh) )
    h′ = @. (1 - α * z) * h + α * z * h̃
    return h′
end

FastGRU(a...; ka...) = FastRecur(FastGRUCell(a...; ka...))
FastRecur(m::FastGRUCell) = FastRecur(m, reduce(hcat, [m.state0 for _ in 1:1]))

# Discrete-time vanilla neural ODE

struct FastNeuralODECell{A,B,C,T,H0,F1,F2,Bool}
    Uh::A
    Wh::B
    f::C
    τ::T
    state0::H0
    hidden_nonlinearity::F1
    dynamics_nonlinearity::F2
    layernorm::Bool
end

function FastNeuralODECell(
    input_dim::Int, 
    hidden_dim::Int;
    dynamics_nn_architecture=Int[],
    τ=1.f0, 
    state0,
    initW=Flux.kaiming_normal, 
    initb=bias_init,
    hidden_nonlinearity=relu,
    dynamics_nonlinearity=tanh,
    layernorm=false
)
    if isempty(dynamics_nn_architecture)
        Uh = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)
        Wh = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)
        f = nothing
    else
        Uh = FastDense( input_dim, dynamics_nn_architecture[1]; initW=initW,  bias=false)
        Wh = FastDense(hidden_dim, dynamics_nn_architecture[1]; initW=initW, initb=initb)
        f = build_nn(
            dynamics_nn_architecture[2:end],
            hidden_nonlinearity,
            dynamics_nn_architecture[1],
            hidden_dim;
            final_nonlinearity = dynamics_nonlinearity,
            layernorm = layernorm,
            initW = initW, 
            initb = initb
        )
    end

    return FastNeuralODECell(
        Uh,
        Wh,
        f,
        τ,
        state0,
        hidden_nonlinearity,
        dynamics_nonlinearity,
        layernorm
    )
end

function (m::FastNeuralODECell)(
    h::AbstractArray, 
    x::AbstractArray, 
    Δt::AbstractFloat,
    params::AbstractVector,  
    lps::Tuple{Vararg{Int}}
)
    α = Δt/m.τ
    Whh, Uhh = m.Wh(h, params[indx(lps,3)]), m.Uh(x, params[indx(lps,2)])
    if isnothing(m.f)
        h̃ = m.layernorm ? ( m.dynamics_nonlinearity.(LN(Whh) .+ Uhh) ) : ( @. m.dynamics_nonlinearity(Whh + Uhh) )
    else
        a_1 = m.layernorm ? ( m.hidden_nonlinearity.(LN(Whh) .+ Uhh) ) : ( @. m.hidden_nonlinearity(Whh + Uhh) )
        h̃ = m.f(a_1, params[indx(lps,4)])
    end
    h′ = @. (1 - α) * h + α * h̃
    return h′
end

FastNeuralODE(a...; ka...) = FastRecur(FastNeuralODECell(a...; ka...))
FastRecur(m::FastNeuralODECell) = FastRecur(m, reduce(hcat, [m.state0 for _ in 1:1]))

# Discrete-time gated neural ODE

struct FastGatedNeuralODECell{A,B,C,D,T,H0,F1,F2,F3,Bool}
    Uh::A
    Wh::B
    f::C
    Uz::A
    Wz::B
    g::D
    τ::T
    state0::H0
    hidden_nonlinearity::F1
    dynamics_nonlinearity::F2
    gating_nonlinearity::F3
    layernorm::Bool
end

function FastGatedNeuralODECell(
    input_dim::Int, 
    hidden_dim::Int;
    dynamics_nn_architecture=Int[],
    gating_nn_architecture=Int[],
    τ=1.f0, 
    state0,
    initW=Flux.kaiming_normal, 
    initb=bias_init,
    hidden_nonlinearity=relu,
    dynamics_nonlinearity=tanh_fast,
    gating_nonlinearity=sigmoid_fast,
    layernorm=false
)
    if isempty(dynamics_nn_architecture)
        Uh = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)
        Wh = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)
        f = nothing
    else
        Uh = FastDense( input_dim, dynamics_nn_architecture[1]; initW=initW,  bias=false)
        Wh = FastDense(hidden_dim, dynamics_nn_architecture[1]; initW=initW, initb=initb)
        f = build_nn(
            dynamics_nn_architecture[2:end],
            hidden_nonlinearity,
            dynamics_nn_architecture[1],
            hidden_dim;
            final_nonlinearity = dynamics_nonlinearity,
            layernorm = layernorm,
            initW=initW, 
            initb=initb
        )
    end

    if isempty(gating_nn_architecture)
        Uz = FastDense( input_dim, hidden_dim; initW=initW,  bias=false)
        Wz = FastDense(hidden_dim, hidden_dim; initW=initW, initb=initb)   
        g = nothing
    else
        Uz = FastDense( input_dim, gating_nn_architecture[1]; initW=initW,  bias=false)
        Wz = FastDense(hidden_dim, gating_nn_architecture[1]; initW=initW, initb=initb)
        g = build_nn(
            gating_nn_architecture[2:end],
            hidden_nonlinearity,
            gating_nn_architecture[1],
            hidden_dim;
            final_nonlinearity = gating_nonlinearity,
            layernorm = layernorm,
            initW=initW, 
            initb=initb
        )
    end

    return FastGatedNeuralODECell(
        Uh,
        Wh,
        f,
        Uz,
        Wz,
        g,
        τ,
        state0,
        hidden_nonlinearity,
        dynamics_nonlinearity,
        gating_nonlinearity,
        layernorm
    )
end

function (m::FastGatedNeuralODECell)(
    h::AbstractArray, 
    x::AbstractArray, 
    Δt::AbstractFloat,
    params::AbstractVector,  
    lps::Tuple{Vararg{Int}}
)
    α = Δt/m.τ
    Wzh, Uzx = m.Wz(h, params[indx(lps,6)]), m.Uz(x, params[indx(lps,5)])
    Whh, Uhh = m.Wh(h, params[indx(lps,3)]), m.Uh(x, params[indx(lps,2)])
    if isnothing(m.g)
        z = m.layernorm ? ( m.gating_nonlinearity.(LN(Wzh) .+ Uzx) ) : ( @. m.gating_nonlinearity(Wzh + Uzx) )
    else
        a_z = m.layernorm ? ( m.hidden_nonlinearity.(LN(Wzh) .+ Uzx) ) : ( @. m.hidden_nonlinearity(Wzh + Uzx) )
        z = m.g(a_z, params[indx(lps,7)])
    end

    if isnothing(m.f)
        h̃ = m.layernorm ? ( m.dynamics_nonlinearity.(LN(Whh) .+ Uhh) ) : ( @. m.dynamics_nonlinearity(Whh + Uhh) )
    else
        a_h = m.layernorm ? ( m.hidden_nonlinearity.(LN(Whh) .+ Uhh) ) : ( @. m.hidden_nonlinearity(Whh + Uhh) )
        h̃ = m.f(a_h, params[indx(lps,4)])
    end

    h′ = @. (1 - α * z) * h + α * z * h̃
    return h′
end

FastGatedNeuralODE(a...; ka...) = FastRecur(FastGatedNeuralODECell(a...; ka...))
FastRecur(m::FastGatedNeuralODECell) = FastRecur(m, reduce(hcat, [m.state0 for _ in 1:1]))

function build_rnn(args, input_dim, h₀)
    this_hidden_nonlinearity = convert_activation_types(args.hidden_nonlinearity)
    this_gating_nonlinearity = convert_activation_types(args.gating_nonlinearity)
    this_dynamics_nonlinearity = convert_activation_types(args.dynamics_nonlinearity)
    initW = convert_init_types(args.init_type, args.dynamics_nn_architecture)
    initb = bias_init

    if args.gru
        rnn = FastGRU(
            input_dim, 
            args.latent_dim; 
            state0=h₀, 
            τ=args.τ, 
            initW=initW,
            initb=initb, 
            dynamics_nonlinearity=this_dynamics_nonlinearity,
            gating_nonlinearity=this_gating_nonlinearity,
            layernorm=args.layernorm
        )
        nns = (rnn.cell.Wz, rnn.cell.Wr, rnn.cell.Wh, rnn.cell.Uz, rnn.cell.Ur, rnn.cell.Uh)
    elseif isa(this_gating_nonlinearity, typeof(identity))
        rnn = FastNeuralODE(
            input_dim, 
            args.latent_dim; 
            dynamics_nn_architecture=args.dynamics_nn_architecture,
            state0=h₀, 
            τ=args.τ, 
            initW=initW,
            initb=initb,
            hidden_nonlinearity=this_hidden_nonlinearity,
            dynamics_nonlinearity=this_dynamics_nonlinearity,
            layernorm = args.layernorm
        )
        nns = (rnn.cell.Uh, rnn.cell.Wh, rnn.cell.f)
    else
        rnn = FastGatedNeuralODE(
            input_dim, 
            args.latent_dim; 
            dynamics_nn_architecture=args.dynamics_nn_architecture,
            gating_nn_architecture=args.gating_nn_architecture,
            state0=h₀, 
            τ=args.τ, 
            initW=initW,
            initb=initb,
            hidden_nonlinearity=this_hidden_nonlinearity,
            dynamics_nonlinearity=this_dynamics_nonlinearity,
            gating_nonlinearity=this_gating_nonlinearity,
            layernorm = args.layernorm
        )
        nns = (rnn.cell.Uh, rnn.cell.Wh, rnn.cell.f, rnn.cell.Uz, rnn.cell.Wz, rnn.cell.g)
    end
    return rnn, nns
end

function build_nn(
    nn_architecture::AbstractVector,
    activation::Function,
    input_dim::Int,
    output_dim::Int;
    initW=Flux.kaiming_normal,
    initb=bias_init,
    layernorm = false,
    final_nonlinearity = identity
)
    function hidden_layer(architecture, layernorm::Bool, i::Int)
        return layernorm ? 
            (FastDense(
                architecture[i],
                architecture[i+1];
                initW=initW, 
                initb=initb
            ), FastLN(activation)) :
            FastDense(
                architecture[i],
                architecture[i+1],
                activation;
                initW=initW, 
                initb=initb
            )
    end
   
    if isempty(nn_architecture)
        nn = layernorm ? 
        FastChain(FastDense(input_dim, output_dim; initW=initW, initb=initb), FastLN(final_nonlinearity)) :
        FastDense(input_dim, output_dim, final_nonlinearity; initW=initW, initb=initb)
    else
        nn_layers = 
            ntuple(
                i -> hidden_layer(nn_architecture, layernorm, i), 
                length(nn_architecture)-1
            )
        nn = layernorm ? 
            FastChain(
                FastDense(
                    input_dim,
                    nn_architecture[1]; 
                    initW=initW, 
                    initb=initb
                ), FastLN(activation),
                tuplejoin(nn_layers...)...,
                FastDense(
                    nn_architecture[end], 
                    output_dim;
                    initW=initW, 
                    initb=initb
                ), FastLN(final_nonlinearity)
            ) :
            FastChain(
                FastDense(
                    input_dim,
                    nn_architecture[1], 
                    activation;
                    initW=initW, 
                    initb=initb
                ), 
                nn_layers...,
                FastDense(
                    nn_architecture[end], 
                    output_dim,
                    final_nonlinearity;
                    initW=initW, 
                    initb=initb
                )
            )
    end

    return nn
end