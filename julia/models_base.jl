using Optim

# %% ==================== Base code for all models ====================

abstract type Model end

Base.length(m::Model) = 1

function preferences(model::Model, t::Trial)
    map(get_data(t)) do datum
        preferences(model, datum)
    end
end

function preferences(model::Model, d::Datum)
    preferences(model, d.b)
end

function mysoftmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function p_rand(q::Vector{Float64})
    n_option = sum(q .> -Inf)
    1 / n_option
end

function logp(prefs::Vector{Float64}, c::Int, α, ε)
    p_soft = mysoftmax(α .* prefs)[c+1]
    if !isfinite(p_soft)
        error("Bad prefs: $prefs")
    end
    p = ε * p_rand(prefs) + (1-ε) * p_soft
    log(p)
end

function c_probs(model::Model, d::Datum, α, ε)
    prefs = preferences(model, d)
    p_soft = mysoftmax(α .* prefs)
    each_p_rand = ε * p_rand(prefs)
    @. each_p_rand * (prefs > -Inf) + (1-ε) * p_soft
end

function logp(model::Model, d::Datum, α::Float64, ε::Float64)
    prefs = preferences(model, d)
    logp(prefs, d.c, α, ε)
end

function logp(model::Model, data::Vector{Datum}, α::Float64, ε::Float64)
    mapreduce(+, data) do d
        logp(model, d, α, ε)
    end
end

function fit_error_model(model::Model, data::Vector{Datum}; x0 = [0.002, 0.1])
    lower = [0., 1e-3]; upper = [5., 1.]
    all_prefs = [preferences(model, d) for d in data]
    cs = [d.c for d in data]
    try
        opt = optimize(lower, upper, x0, Fminbox(LBFGS())) do (α, ε)
            - mapreduce(+, all_prefs, cs) do prefs, c
                logp(prefs, c, α, ε)
            end
        end
        (α=opt.minimizer[1], ε=opt.minimizer[2], logp=-opt.minimum)
    catch
        (α=-1., ε=1., logp=NaN)
    end
end

function fit_model(model_class::Type, trials::Vector{Trial})
    fit_model(model_class, get_data(trials))
end

function fit_model(model_class::Type, data::Vector{Datum})
    err_fits = map(instantiations(model_class)) do model
        fit = fit_error_model(model, data)
        (model=model, fit...)
    end
    best = argmax([ef.logp for ef in err_fits])
    return err_fits[best]
end

Fit = NamedTuple{(:model, :α, :ε, :logp)}

function logp(fit::Fit, d::Datum)
    prefs = preferences(fit.model, d)
    logp(prefs, d.c, fit.α, fit.ε)
end

