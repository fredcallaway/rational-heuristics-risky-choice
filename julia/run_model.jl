using Serialization
using CSV
using Distributed

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("utils.jl")
    include("data.jl")
end
mkpath("tmp/policies")

# define MetaMDP for each condition
version = "1.0"
pdf = CSV.File("../data/processed/$version/participants.csv");
conditions = pdf |> map(x -> (alpha=x.alpha, sigma=x.sigma, cost=x.cost)) |> unique
all_mdps = map(conditions) do cond
    MetaMDP(
        reward_dist=Normal(0, cond.sigma),
        weight_dist=Dirichlet(ones(4) .* cond.alpha),
        cost=cond.cost
    )
end |> collect;

# set parameters for bayesian optimization
@everywhere opt_kws = (seed=1, n_iter=500, n_roll=10000, α=100)
serialize("tmp/opt_kws", opt_kws)

# optimize!
@time pmap(enumerate(all_mdps)) do (i, m)
    policy = optimize_bmps(m; opt_kws..., verbose=false)
    serialize("tmp/policies/$i", policy)
    policy
end

@everywhere function simulate(pol::Policy)::Trial
    s = State(pol.m)
    uncovered = Int[]
    roll = rollout(pol, s) do b, c
        c != ⊥ && push!(uncovered, c)
    end
    Trial(
        hash(pol.m) % 100000,
        0,  # trial_index
        pol.m.reward_dist.σ,  # sigma
        pol.m.weight_dist.alpha[1],  # alpha
        pol.m.cost,  # cost
        s.weights,  # weights
        s.matrix ./ s.weights,  # values
        uncovered,  # uncovered
        sample(Weights(choice_probs(roll.belief)))  # choice
    )
end

# simulate!
mkpath("tmp/sims")
@time pmap(eachindex(all_mdps)) do i
    pol = deserialize("tmp/policies/$i")
    sims = map(1:100000) do i
        simulate(pol)
    end
    serialize("tmp/sims/$i", sims)
end;

