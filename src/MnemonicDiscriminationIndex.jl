module MnemonicDiscriminationIndex

using Distributions: Uniform
using LsqFit: curve_fit
using PrecompileTools: @setup_workload, @compile_workload
using QuadGK: quadgk
using Random

struct MDIResult{T,D}
    params::Vector{T}
    auc::T
    startval::T
    endval::T
    domain::Tuple{D, D}
    Δ::T
    λ::T
end

function _fit_model(model, dissimilarities, responses, p0; ntries=10000, kwargs...)
    n_errs = 0
    # Try again with different initial parameter values until curve_fit returns
    while true
        try
            return curve_fit(model, dissimilarities, responses, p0(); kwargs...)
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

const DOMAIN = (0, 1)

"""
    fit_model(model, dissimilarities, responses, p0; [domain=(0,1), kwargs...])

Fit the model to the data and return an `MDIResult` struct.

`p0` is a function that generates initial parameters relevant for the passed-in `model`.

The `domain` argument is a tuple with the lowest and highest values of dissimilarity. Should not typically be changed.

The `kwargs` get passed on to `curve_fit`.
"""
function fit_model(model, dissimilarities, responses, p0; domain=DOMAIN, kwargs...)
    params = _fit_model(model, dissimilarities, responses, p0; kwargs...).param
    return fit_model(model, params; domain)
end

function fit_model(model, params; domain=DOMAIN)
    startval = model(domain[1], params)
    endval = model(domain[2], params)
    auc, = quadgk((x) -> endval - model(x, params), domain[1], domain[2])
    Δ = endval - startval
    λ = 1 - auc / Δ

    return MDIResult(params, auc, startval, endval, float.(domain), Δ, λ)
end

export fit_model, MDIResult

include("logistic5.jl")

export logistic5, fit_logistic5

## Precompilation
@setup_workload begin
    old_or_new = [0, 0, 0, 1, 0, 1, 1, 1]
    dissimilarities = 0:(1/7):1

    @compile_workload begin
        logistic5_results = fit_logistic5(dissimilarities, old_or_new)
    end
end

end # module MnemonicDiscriminationIndex
