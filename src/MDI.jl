module MDI

using Distributions
using LsqFit
using QuadGK
using Random
using StableRNGs

# This is the function that we will try to fit
@. logistic5(data,params) = params[4]+(params[1]-params[4])/((1+(data/params[3])^params[2])^params[5])

# Randomly initialize the initial parameters to help prevent non-convergence
function p_init(rng)
    a = rand(rng, Uniform(0, 0.1))
    b = rand(rng, Uniform(1, 10))
    c = rand(rng, Uniform(0.1, 0.7))
    d = rand(rng, Uniform(0.9, 1))
    e = rand(rng, Uniform(0.5, 1.5))
    return [a, b, c, d, e]
end

# Tries to fit the function until it succeeds
function fit_model(data_x, data_y; model=logistic5, lower=[0.,0,0,0,0], upper=[1.,Inf,1,1,Inf], seed::Union{Nothing,Int}=nothing, kwargs...)
    rng = isnothing(seed) ? Random.default_rng() : StableRNG(seed)

    # Try again with different initial parameter values until curve_fit returns
    while true
        try
            return curve_fit(model,data_x,data_y, p_init(rng); lower, upper, kwargs...)
        catch
            continue
        end
    end
end

# Generates the function to be used to calculate the area difference
function get_area_diff(max, params; model=logistic5)
    return area_diff(x) = max - model(x, params)
end

# Returns the auc and the properly scaled auc
function get_aucs(params; domain=(0,1), model=logistic5)
    min = model(domain[1], params)
    max = model(domain[2], params)
    area_diff = get_area_diff(max, params; model)
    auc, = quadgk(area_diff, domain[1], domain[2])
    auc_scaled = auc/(max-min)
    return auc, auc_scaled, min, max
end


export logistic5, fit_model, get_aucs

end # module MDI
