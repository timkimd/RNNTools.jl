function bias_init(dims...)
    Flux.randn32(dims...) .* 0.001f0
end

function our_normal(dims...)
    Flux.kaiming_normal(dims...; gain = 1.f0)
end

indx(lps, n) = n > 1 ? (lps[n-1]+1:lps[n]) : (1:lps[1])

"""
    funs_find_indx(nns::Tuple{Vararg{N}}, lps::Tuple{Vararg{Int}}) where {N}

Description.

ARGUMENT
-`nns`: 
-`lps`: 

RETURN
-
"""

function funs_find_indx(nns::Tuple{Vararg{N}}, lps::Tuple{Vararg{Int}}) where {N}
    return ntuple(i-> ((x, p) -> nns[i](x, p[indx(lps, i)])), length(nns))
end

"""
    tuplejoin
"""
tuplejoin(x) = x
tuplejoin(x, y) = (x..., y...)
tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

"""
    split_data(data, obs_dim, tmin, tmax, tsteps)

Description.

ARGUMENT
-`data`: 
-`obs_dim`: 

RETURN
-
"""

function split_data(data, obs_dim)
    observations = data[1:obs_dim, :, :]
    if size(data, 1) == 2*obs_dim
        inputs = nothing
        masks = data[obs_dim+1:end, :,  :] .> 0
    else
        inputs = data[(obs_dim+1):(size(data, 1)÷2), :, :]
        masks = data[(size(data, 1)÷2)+1:(size(data, 1)÷2)+obs_dim, :, :] .> 0
    end
    return observations, inputs, masks
end


"""
    concat_data(observations, inputs=nothing, masks=nothing)

Description.

ARGUMENT
-`observations`: 
-`inputs` (optional): 
-`masks` (optional): 

RETURN
-
"""

function concat_data(observations, inputs=nothing, masks=nothing)
    if isnothing(masks)
        if isnothing(inputs)
            observations_ = vcat(
                observations, 
                trues(
                    size(
                        observations
                    )
                )
            )
        else
            observations_ = vcat(
                observations, 
                inputs,
                trues(
                    size(
                        observations
                    )
                ),
                trues(
                    size(
                        inputs
                    )
                )
            )
        end
    else
        if isnothing(inputs)
            observations_ = vcat(
                masks .* observations, 
                masks
            )
        else
            observations_ = vcat(
                masks .* observations, 
                inputs,
                masks,
                trues(
                    size(
                        inputs
                    )
                )
            )
        end
    end
    return observations_
end

"""
    convert_init_types(args_init)

Description.

ARGUMENT
-`args_init`: 

RETURN
-
"""

function convert_init_types(args_init, args_architecture)
    if args_init == "glorot_normal"
        initW = Flux.glorot_normal
    elseif args_init == "glorot_uniform"
        initW = Flux.glorot_uniform
    elseif args_init == "kaiming_normal"
        initW = Flux.kaiming_normal
    elseif args_init == "kaiming_uniform"
        initW = Flux.kaiming_uniform
    else
        initW = our_normal
    end
    return initW
end


"""
    convert_solver_types(args_solver)

Description.

ARGUMENT
-`args_solver`: 

RETURN
-
"""

function convert_solver_types(args_solver)
    if args_solver == "Euler()"
        solver = Euler()
    elseif args_solver == "Tsit5()" # for most non-stiff problems
        solver = Tsit5()
    elseif args_solver == "AutoTsit5(TRBDF2(autodiff=false))" #with a stiffness detection and auto-switching algorithm
        solver = AutoTsit5(TRBDF2(autodiff=false))
    elseif args_solver == "BS3()" # for fast solving at higher tolerances
        solver = BS3()
    elseif args_solver == "Vern7()" # for high accuracy but with the range of Float64 (~1e-8-1e-12)
        solver = Vern7()
    elseif args_solver == "VCABM()" # for high accuracy when the system of equations is very large (>1,000 ODEs)
        solver = VCABM()
    elseif args_solver == "DP5()" # dopri5, used in many neural ODE papers out there
        solver = DP5()
    end
    return solver
end

"""
    convert_sensealg_types(args_sensealg)

Description.

ARGUMENT
-`args_sensealg`: 

RETURN
-
"""

function convert_sensealg_types(args_sensealg)
    if args_sensealg == "default"
        sensealg = nothing
    elseif args_sensealg == "BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))"
        sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)) # does not support GPUs
    elseif args_sensealg == "InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))"
        sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) # does not support GPUs
    elseif args_sensealg == "BacksolveAdjoint(autojacvec=ZygoteVJP())"
        sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP())
    elseif args_sensealg == "BacksolveAdjoint(autojacvec=TrackerVJP())"
        sensealg = BacksolveAdjoint(autojacvec=TrackerVJP()) # this should be *the* choice of adjoint for discrete inputs
    elseif args_sensealg == "InterpolatingAdjoint(autojacvec=ZygoteVJP())"
        sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    end

    return sensealg
end

"""
    convert_interp_types(args_interp)

Description.

ARGUMENT
-`args_interp`: 

RETURN
-
"""

function convert_interp_types(args_interp)
    if args_interp == "Constant"
        interp = ConstantInterpolation
    elseif args_interp == "Linear"
        interp = LinearInterpolation
    elseif args_interp == "Quadratic"
        interp = QuadraticInterpolation
    elseif args_interp == "Lagrange"
        interp = LagrangeInterpolation
    elseif args_interp == "CubicSpline"
        interp = CubicSpline
    elseif args_interp == "BSpline"
        interp = BSplineInterpolation
    elseif args_interp == "BSplineApprox"
        interp = BSplineApprox
    end

    return interp
end

"""
    convert_activation_types(args_activation)

Description.

ARGUMENT
-`args_activation`: 

RETURN
-
"""

function convert_activation_types(args_activation)
    if args_activation == "identity"
        activation = identity
    elseif args_activation == "sigmoid"
        activation = sigmoid_fast
    elseif args_activation == "sigmoid_beta"
        activation = σᵦ
    elseif args_activation == "swish"
        activation = swish
    elseif args_activation == "relu"
        activation = relu
    elseif args_activation == "gelu"
        activation = gelu
    elseif args_activation == "tanh"
        activation = tanh_fast
    end
    return activation
end


"""
    get_dynamics_fun(args; inputs=nothing, params)

Description.

ARGUMENT
-`args`: 
-`inputs` (optional): 
-`params`: 

RETURN
-
"""

function get_dynamics_fun(args; inputs=nothing, params)
    this_activation = convert_activation_types(args.activation)
    input_dim = isnothing(inputs) ? 0 : size(inputs, 1)

    ƒ = build_nn(
        args.dynamics_nn_architecture,
        this_activation,
        args.latent_dim + input_dim,
        args.latent_dim
    )

    θ = params[1:length(initial_params(ƒ))]

    function F(u)
        # du/dt = F(u)
        ƒ(u,θ)
    end

    function F_w_input(u, ext)
        # du/dt = F(u, input)        
        ƒ(vcat(u, ext), θ)
    end

    if isnothing(inputs)
        return F
    else
        return F_w_input
    end
end


division((s,n)) = s/n
normalize_std((s,n)) = s / (n-1)
normalize_stderr((s,n)) = s / (n * (n-1))
misssum_elem(s, x) = (x===missing || isnan(x)) ? s : s+x
misssum_tuple((s,n), x) = (x===missing || isnan(x)) ? (s,n) : (s+x, n+1)


"""
    misssum(a;dims=:)

sum that respects `missing` or `NaN`.

only works with 3D objects.

"""

misssum(a;dims=:) = reduce(misssum_elem, a, init = zero(eltype(a)), dims=dims)


"""
    missmean(a;dims=:)

mean that respects `missing` or `NaN`.

"""

missmean(a;dims=:) = division.(reduce(misssum_tuple, a, init = (zero(eltype(a)), 0), dims=dims))


"""
    missstd(a;dims=:)

std that respects `missing` or `NaN`.

"""

missstd(a;dims=:) = sqrt.(normalize_std.(reduce(misssum_tuple, (a .- missmean(a;dims=dims)).^2, init = (zero(eltype(a)), 0), dims=dims)))


"""
    missstderr(a;dims=:)

stderr that respects `missing` or `NaN`.

"""

missstderr(a;dims=:) = sqrt.(normalize_stderr.(reduce(misssum_tuple, (a .- missmean(a;dims=dims)).^2, init = (zero(eltype(a)), 0), dims=dims)))

"""
    initial_params(nns)

Description.

ARGUMENT
-`nns`: 

RETURN
-
"""

DiffEqFlux.initial_params(f::Vector{<:AbstractFloat}) = f

function get_initial_params(nns)
    function get_initial_params_(i::Int)
        return initial_params(nns[i])
    end
    return ntuple(get_initial_params_, length(nns))
end

function closest_index(x, val)
    ibest = first(eachindex(x))
    dxbest = abs(x[ibest]-val)
    for i in eachindex(x)
        dx = abs(x[i]-val)
        if dx < dxbest
            dxbest = dx
            ibest = i
        end
    end
    return ibest
end

function myConstantInterpolation(data, saveat, t)
    indx = closest_index(saveat, t)
    T = size(data, 3)
    if T >= indx > 0
        return @view data[:, :, indx]
    elseif indx <= 0
        return @view data[:, :, 1]
    elseif indx > T
        return @view data[:, :, end]
    end
end


"""
    train_val_test_split(args, data; seed)

Description.

ARGUMENT
-`args`: 
-`data`: 
-`seed`:

RETURN
-
"""

function train_val_test_split(args, data; seed)
end


"""
    compute_R²(pred, data)

Compute the coefficient of determination (R²).

ARGUMENT
-`pred`: vector with model predictions
-`data`: vector with data

RETURN
-A scalar coefficient of determination
"""

function compute_R²(pred, data)
    @assert length(pred) == length(data)
    SS_tot = sum(abs2, data .- Statistics.mean(data))
    SS_res = sum(abs2, data .- pred)
    R² = 1 - (SS_res / SS_tot)
    return R²
end


"""
Based on MATLAB's meshgrid

X,Y = meshgrid(x,y) returns 2-D grid coordinates based on the coordinates contained in vectors x and y. 
X is a matrix where each row is a copy of x, and Y is a matrix where each column is a copy of y. 
The grid represented by the coordinates X and Y has length(y) rows and length(x) columns.

X,Y,Z = meshgrid(x,y,z) returns 3-D grid coordinates based on the coordinates contained in vectors x, y and z.
"""

function meshgrid(
    x::AbstractVector{T}, 
    y::AbstractVector{T}
) where {T <: Real}

    m, n = length(y), length(x)
    x = reshape(x, 1, n)
    y = reshape(y, m, 1)

    return (
        repeat(x, m, 1), 
        repeat(y, 1, n)
    )
end

function meshgrid(
    x::AbstractVector{T}, 
    y::AbstractVector{T}, 
    z::AbstractVector{T}
) where {T <: Real}

    l, m, n = length(z), length(y), length(x)
    x = reshape(x, 1, 1, n)
    y = reshape(y, 1, m, 1)
    z = reshape(z, l, 1, 1)

    return (
        repeat(x, l, m, 1), 
        repeat(y, l, 1, n), 
        repeat(z, 1, m, n)
    )
end