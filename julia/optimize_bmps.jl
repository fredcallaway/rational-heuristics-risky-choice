using Distributed
using Printf
using Sobol
using Statistics
include("gp_min.jl")

const TOO_EXPENSIVE = -100.

function max_cost(m::MetaMDP)
    θ = [1., 0, 0, 1]
    # s = State(m)
    # b = Belief(s)
    function computes()
        pol = BMPSPolicy(m, θ)
        any(act(pol, Belief(m); check_voc1=false) != ⊥ for i in 1:10)
    end

    while computes()
        θ[1] *= 2
        @debug "Increasing" θ[1]
    end

    while !computes()
        θ[1] /= 2
        @debug "Decreasing" θ[1]
        if θ[1] < 2^-10
            return TOO_EXPENSIVE
        end
    end

    step_size = θ[1] / 10
    while computes()
        θ[1] += step_size
        @debug "Increasing" θ[1]
    end
    θ[1]
end


function x2theta(mc, x)
    voi_weights = diff([0; sort(collect(x[2:3])); 1])
    [x[1] * mc; voi_weights]
end


function mean_reward(policy, n_roll, parallel)
    if parallel
        rr = @distributed (+) for i in 1:n_roll
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    else
        rr = mapreduce(+, 1:n_roll) do i
            rollout(policy, max_steps=200).reward
        end
        return rr / n_roll
    end

end

function optimize_bmps(m::MetaMDP; α=Inf, n_iter=500, seed=nothing, n_roll=10000,
                  verbose=false, parallel=true, repetitions=1)
    if seed != nothing
        Random.seed!(seed)
    end
    mc = max_cost(m)
    if mc == TOO_EXPENSIVE
        println("Computation too expensive, skipping optimization.")
        return BMPSPolicy(m, [1., 0, 0, 0])
    end


    function loss(x, nr=n_roll)
        policy = BMPSPolicy(m, x2theta(mc, x), α)
        reward, secs = @timed mean_reward(policy, n_roll, parallel)
        if verbose
            print("θ = ", round.(x2theta(mc, x); digits=2), "   ")
            @printf "reward = %.3f   seconds = %.3f\n" reward secs
            flush(stdout)
        end
        -reward
    end

    opt = gp_minimize(loss, 3, noisebounds=[-4, -2],
                      iterations=n_iter, repetitions=repetitions,
                      verbose=false)

    f_mod = loss(opt.model_optimizer, 10 * n_roll)
    f_obs = loss(opt.observed_optimizer, 10 * n_roll)
    best = f_obs < f_mod ? opt.observed_optimizer : opt.model_optimizer
    return BMPSPolicy(m, x2theta(mc, best), α)
end



