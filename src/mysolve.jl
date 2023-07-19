# adapted from GalacticOptim.jl
function mysolve(
    f::Function,
    ps::AbstractVector, 
    opt::Flux.Optimise.AbstractOptimiser, 
    data::NCycle{MyDataLoader{D, R}}, 
    device;
    cb = (args...) -> (false),
    save_best = false, 
    kwargs...
) where {D, R<:AbstractRNG}
    maxiters = length(data)

    θ = copy(ps)

    local x, min_err, min_θ, nfe
    min_err = typemax(eltype(ps)) #dummy variables
    min_opt = 1
    min_θ = ps

    for (i,d_) in enumerate(data)
      local d = d_ |> device
      G = Zygote.gradient(x -> f(x, d)[1], θ)[1]
      x, _, _, nfe = f(θ, d)
      cb_call = cb(θ, G, x, nfe)
      if save_best
        if x < min_err  #found a better solution
          min_opt = opt
          min_err = x
          min_θ = copy(θ)
        end
        if i == maxiters  #Last iteration, revert to best.
          opt = min_opt
          x = min_err
          θ = min_θ
          cb(θ, G, x, nfe)
          break
        end
      end
      Flux.update!(opt, θ, G)
      d = nothing
      GC.gc(true)
    end

    return θ # minimizer
end