include("meta_mdp.jl")
include("bmps.jl")
using JSON
using LinearAlgebra

# struct RoundDirichlet <: Distribution{Multivariate,Continuous}
#     d::Dirichlet
#     digits::Int
# end
# RoundDirichlet(d::Dirichlet) = RoundDirichlet(d, 1)
# RoundDirichlet(α::Vector{Float64}) = RoundDirichlet(Dirichlet(α), 1)

# Base.rand(d::RoundDirichlet) = round.(d.d; digits=d.digits)

# %% ====================  ====================
n_gamble = 4
n_outcome = 2
n = n_gamble * n_outcome
idx = reshape(1:n, n_outcome, n_gamble)

σ = 100
V = collect(Diagonal(σ^2  * ones(n)))

for col in eachcol(idx)
    for (i, j) in Iterators.product(col, col)
        if i ≠ j
            V[i, j] = .9 * σ^2
        end
    end
end


reshape(rand(MvNormal(zeros(n), V)), n_outcome, n_gamble)

# %% ====================  ====================

m = MetaMDP(
    n_gamble=6,
    n_outcome=4,
    # reward_dist=Normal(20, 10),
    reward_dist=Normal(0, 1),
    weight_dist=Dirichlet(ones(4)),
    cost=1
)

# pol = MetaGreedy(m, Inf)
# # pol = BMPSPolicy(m, (cost = 0.007031250000000003, voi1 = 0.5546875, voi_action = 0.109375, vpi = 0.3359375), Inf)
# 1000 \ mapreduce(+, 1:1000) do i
#     x = rollout(pol)
#     [x.reward, x.steps]
# end

# %% ====================  ====================


function make_trial(m::MetaMDP)
    w = nothing
    while true
        w = round.(rand(m.weight_dist); digits=2)
        w[1] += 1 - sum(w)
        all(w .> 0) && break
    end

    X = rand(m.reward_dist, (m.n_outcome, m.n_gamble))
    # X = round.(Int, rand(m.reward_dist, (m.n_outcome, m.n_gamble)))
    # X = max.(X, 0)
    (payoff_matrix=X', probabilities=w, id=randstring(15))
end

"A trial where one gamble dominates."
function dominating_trial(m::MetaMDP)
    t = make_trial(m)
    M = t.payoff_matrix
    ng, no = size(M)
    g = rand(1:ng)
    for o in 1:no
        M[g,o] = -1000000  # don't count towards maximum below
        M[g,o] = maximum(M[:,o]) + rand(Uniform(0.1, 0.5))
    end
    t
end

include("box.jl")
αs = logscale.(0:0.25:1, 0.1, 10)
# costs = [0,1,2,4,8]

trial_data = map(enumerate(αs)) do (i, α)
    m = MetaMDP(
        n_gamble=6,
        n_outcome=4,
        # reward_dist=Normal(20, 10),
        reward_dist=Normal(0, 1),
        weight_dist=Dirichlet(α .* ones(4)),
        cost=1
    )

    data = (
        params = (mu=m.reward_dist.μ, sigma=m.reward_dist.σ, alpha=m.weight_dist.alpha[1]),
        standard = [make_trial(m) for i in 1:100],
        dominating = [dominating_trial(m) for i in 1:30],
    )
    open("/Users/fred/heroku/mouselab/static/json/trials_$i.json", "w") do f
        write(f, JSON.json(data))
    end
    return data
end;



