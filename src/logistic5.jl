"""
  `logistic5(data, params)`

  Calculates the value(s) at the point(s) in `data` for the logistic5 curve with `params`.

  See: Cardillo G. (2012) Five parameters logistic regression - There and back again http://www.mathworks.com/matlabcentral/fileexchange/38043
"""
@. logistic5(data,params) = params[4]+(params[1]-params[4])/((1+(data/params[3])^params[2])^params[5])

function _p0_logistic5(rng)
    a = rand(rng, Uniform(0, 0.1))
    b = rand(rng, Uniform(1, 10))
    c = rand(rng, Uniform(0.1, 0.7))
    d = rand(rng, Uniform(0.9, 1))
    e = rand(rng, Uniform(0.5, 1.5))
    return [a, b, c, d, e]
end

"""
  `p0_logistic5(rng)`

  Randomly initialize the initial logistic5 parameters to help prevent non-convergence.
"""
p0_logistic5(rng) = () -> _p0_logistic5(rng)

"""
  `fit_logistic5(data_x, data_y;lower=Float64[0,0,0,0,0], upper=Float64[1,Inf,1,1,Inf], rng=Random.default_rng(), kwargs...)`

  Wrapper around `fit_model` for the `logistic5` function

  The `kwargs` get passed on to `curve_fit`.
"""
fit_logistic5(data_x, data_y;lower=Float64[0,0,0,0,0], upper=Float64[1,Inf,1,1,Inf], rng=Random.default_rng(), kwargs...) =
    fit_model(logistic5, data_x, data_y, p0_logistic5(rng); lower, upper, kwargs...)
