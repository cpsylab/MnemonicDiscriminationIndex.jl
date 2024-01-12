module MDI

using Distributions: Uniform
using LsqFit: curve_fit
using QuadGK: quadgk
using Random

"""
  `fit_model(model, data_x, data_y, p0; kwargs...)`

  Wrapper around `LsqFit`'s `curve_fit` that tries fitting the curve to the `model` until it succeeds.

  `p0` is a function that generates initial parameters relevant for the passed-in `model`.

  The `kwargs` get passed on to `curve_fit`.
"""
function fit_model(model, data_x, data_y, p0; kwargs...)
    # Try again with different initial parameter values until curve_fit returns
    while true
        try
            return curve_fit(model,data_x,data_y, p0(); kwargs...)
        catch e
            if e isa InexactError
                continue
            else
                rethrow()
            end
        end
    end
end

struct AUC{T}
    auc::T
    startval::T
    endval::T
    domain
end

"""
  `get_aucs(model, params; [domain=(0,1)])`

  Returns an `AUC` struct containing the `auc`, the `startval`, and the `endval`.

  This has not been tested with a different `domain`. Change it at your own risk!
"""
function get_aucs(model, params; domain=(0,1))
    startval = model(domain[1], params)
    endval = model(domain[2], params)
    auc, = quadgk((x) -> endval - model(x, params), domain[1], domain[2])
    return AUC(auc, startval, endval, domain)
end

struct MDIndices{T}
    Δ::T
    λ::T
end

"""
  `get_MD_indices(auc::AUC)`

  Returns an `MDIndices` struct containing the Δ and λ indices of `auc`.
"""
function get_MD_indices(auc::AUC)
    Δ = auc.endval - auc.startval
    return MDIndices(Δ, auc.auc/Δ)
end

export fit_model, get_aucs, get_MD_indices

include("logistic5.jl")

export logistic5, fit_logistic5

end # module MDI
