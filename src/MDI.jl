module MDI

using Distributions: Uniform
using LsqFit
using QuadGK
using Random

"""
  `logistic5(data, params)`

  Calculates the value(s) at the point(s) in `data` for the logistic5 curve with `params`.

  See: Cardillo G. (2012) Five parameters logistic regression - There and back again http://www.mathworks.com/matlabcentral/fileexchange/38043
"""
@. logistic5(data,params) = params[4]+(params[1]-params[4])/((1+(data/params[3])^params[2])^params[5])

"""
  `_p_init(rng)`

  Randomly initialize the initial logistic5 parameters to help prevent non-convergence.
"""
function _p_init(rng)
    a = rand(rng, Uniform(0, 0.1))
    b = rand(rng, Uniform(1, 10))
    c = rand(rng, Uniform(0.1, 0.7))
    d = rand(rng, Uniform(0.9, 1))
    e = rand(rng, Uniform(0.5, 1.5))
    return [a, b, c, d, e]
end

"""
  `fit_model(data_x, data_y; model=logistic5, lower=[0.,0,0,0,0], upper=[1.,Inf,1,1,Inf], seed::Union{Nothing,Int}=nothing, kwargs...)`

  Wrapper around `LsqFit`'s `curve_fit` that tries fitting the curve to the `model` until it succeeds.

  The `kwargs` get passed on to `curve_fit`.
"""
function fit_model(data_x, data_y; model=logistic5, lower=Float64[0,0,0,0,0], upper=Float64[1,Inf,1,1,Inf], rng=Random.default_rng(), kwargs...)
    # Try again with different initial parameter values until curve_fit returns
    while true
        try
            return curve_fit(model,data_x,data_y, _p_init(rng); lower, upper, kwargs...)
        catch e
            if e isa InexactError
                continue
            else
                rethrow()
            end
        end
    end
end

"""
  `_get_area_diff(endval, params; model=logistic5)`

  Generates the function to be used to calculate the area difference.
"""
function _get_area_diff(endval, params; model=logistic5)
    return area_diff(x) = endval - model(x, params)
end

"""
  `get_aucs(params; [model=logistic5], [domain=(0,1)])`

  Returns the auc, the properly scaled auc, and the start and end values of the `model` given the `params`.
"""
function get_aucs(params; model=logistic5, domain=(0,1))
    startval = model(domain[1], params)
    endval = model(domain[2], params)
    area_diff = _get_area_diff(endval, params; model)
    auc, = quadgk(area_diff, domain[1], domain[2])
    auc_scaled = auc/(endval-startval)
    return auc, auc_scaled, startval, endval
end

export logistic5, fit_model, get_aucs

end # module MDI
