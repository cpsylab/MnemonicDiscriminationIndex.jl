module MnemonicDiscriminationIndex

using Distributions: Uniform
using LsqFit: curve_fit
using PrecompileTools: @setup_workload, @compile_workload
using QuadGK: quadgk
using Random

"""
  `fit_model(model, data_x, data_y, p0; kwargs...)`

  Wrapper around `LsqFit`'s `curve_fit` that tries fitting the curve to the `model` until it succeeds.

  `p0` is a function that generates initial parameters relevant for the passed-in `model`.

  The `kwargs` get passed on to `curve_fit`.
"""
function fit_model(model, data_x, data_y, p0; ntries=10000, kwargs...)
    n_errs = 0
    # Try again with different initial parameter values until curve_fit returns
    while true
        try
            return curve_fit(model, data_x, data_y, p0(); kwargs...)
        catch e
            # These errors can happen through bad luck of the initial parameters
            if e isa DomainError || e isa ArgumentError || e isa InexactError
                n_errs += 1

                # Keep going as long as the error hasn't happened too many times
                n_errs < ntries && continue

                # If one of these errors has been thrown too many times,
                #  it's probably a user-caused ArgumentError
                @error "curve_fit has errored $(n_errs) times. Giving up."
            end
            # Always rethrow any other errors
            rethrow()
        end
    end
end

struct AUC{T}
    auc::T
    startval::T
    endval::T
    domain::Any
end

"""
  `get_auc(model=logistic5, params; [domain=(0,1)])`

  Returns an `AUC` struct containing the `auc`, the `startval`, and the `endval`.

  This has not been tested with a different `domain`. Change it at your own risk!
"""
function get_auc(model, params; domain=(0, 1))
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

  Returns an `MDIndices` struct containing the `Δ` and `λ` indices of `auc`.
"""
function get_MD_indices(auc::AUC)
    Δ = auc.endval - auc.startval
    return MDIndices(Δ, 1 - auc.auc / Δ)
end

export fit_model, get_auc, get_MD_indices

include("logistic5.jl")

export logistic5, fit_logistic5

## Precompilation
@setup_workload begin
    old_or_new = [0, 0, 0, 1, 0, 1, 1, 1]
    distance = 0:(1/7):1

    @compile_workload begin
        logistic5_params = fit_logistic5(distance, old_or_new).param
        auc = get_auc(logistic5_params)
        mdis = get_MD_indices(auc)
    end
end

end # module MnemonicDiscriminationIndex