using Serialization
using CSV
using Distributed
using ProgressMeter

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("utils.jl")
    include("data.jl")
end
mkpath("tmp/policies")

@everywhere id(m::MetaMDP) = join([round(m.weight_dist.alpha[1]; digits=3), m.reward_dist.σ, m.cost], "-")

# %% ==================== setup ====================

version = "1.0"
implicit_costs = [0:0.1:3; 4:17]

participants = CSV.File("../data/processed/$version/participants.csv");
alphas = participants |> map(x -> x.alpha) |> unique |> sort
sigmas = participants |> map(x -> x.sigma) |> unique |> sort
explicit_costs = participants |> map(x -> x.cost) |> unique |> sort
costs = map(sum, Iterators.product(explicit_costs, implicit_costs)) |> unique |> sort

all_mdps = map(Iterators.product(alphas, sigmas, costs)) do (alpha, sigma, cost)
    MetaMDP(
        reward_dist=Normal(0, sigma),
        weight_dist=Dirichlet(ones(4) .* alpha),
        cost=cost
    )
end |> collect;

# %% ==================== policy optimization ====================

# set parameters for bayesian optimization
@everywhere opt_kws = (seed=1, n_iter=500, n_roll=10000, α=100)
serialize("tmp/opt_kws", opt_kws)

# optimize! (this takes a frustratingly long time because of a bug in the optimizer)
mkpath("tmp/policies")
policies = @showprogress pmap(all_mdps; retry_delays=zeros(100)) do m
    try
        cache("tmp/policies/" * id(m)) do
            optimize_bmps(m; opt_kws..., verbose=false)
        end
    catch err
        @error "Optimization error!" err id(m)
        rethrow()
    end
end

# %% ==================== simulation ====================

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
        # pol.m.cost,  # cost
        0,
        s.weights,  # weights
        s.matrix ./ s.weights,  # values
        uncovered,  # uncovered
        sample(Weights(choice_probs(roll.belief)))  # choice
    )
end


# first write those that aren't computed
mkpath("tmp/sims")
@showprogress pmap(policies) do pol
    isfile("tmp/sims/" * id(pol.m)) && return 0
    sims = map(1:100000) do i
        simulate(pol)
    end
    serialize("tmp/sims/" * id(pol.m), sims)
    return 1
end

# then load them all (two steps because data transfer contributes substantially to runtime)
all_sims = @showprogress map(policies) do pol
    deserialize("tmp/sims/" * id(pol.m))
end;

# %% ==================== find best cost ====================
dictionary = SplitApplyCombine.dictionary

condition(t::Trial) = map(float, (t.sigma, t.alpha, t.cost))
condition(m::MetaMDP) = (m.reward_dist.σ, m.weight_dist.alpha[1], m.cost)

all_trials = load_trials(version);
human = map(group(condition, all_trials)) do trials
    # length(trials)
    mean(length(t.uncovered) for t in trials)
end

grouped_sims = map(all_mdps, all_sims) do m, trials
    condition(m) => trials
end |> dictionary

model = map(grouped_sims) do trials
    mean(length(t.uncovered) for t in trials)
end

# %% --------

best_implicit_cost = argmin(implicit_costs) do implicit_cost
    map(pairs(human)) do ((sigma, alpha, cost), h)
        m = model[(sigma, alpha, cost + implicit_cost)]
        # (h - m) ^ 2
        # abs(h - m)
        h - m
    end |> sum |> abs
end

# %% --------

map(pairs(human)) do ((sigma, alpha, cost), h)
    argmin(implicit_costs) do implicit_cost
        m = model[(sigma, alpha, cost + implicit_cost)]
        # (h - m) ^ 2
        # abs(h - m)
        abs(h - m)
    end
end

# %% --------

best_by_cost = map(explicit_costs) do cost
    argmin(implicit_costs) do implicit_cost
        loss = map(pairs(human)) do ((sigma, alpha, cost2), h)
            cost != cost2 && return 0.
            m = model[(sigma, alpha, cost + implicit_cost)]
            h - m
        end |> sum |> abs
    end
end

# %% --------
best_implicit_cost = best_by_cost[1]
mkpath("results/sims/$best_implicit_cost")
@showprogress for (sigma, alpha, cost) in keys(human)
    sims = grouped_sims[(sigma, alpha, cost + best_implicit_cost)][1:10000]
    sims = map(sims) do t
        mutate(t, cost=cost)
    end
    name = join([sigma, alpha, cost], "-")
    write("results/sims/$best_implicit_cost/$name.json", JSON.json(sims))
end

# %% ==================== implicit cost vs reward ====================


cost_vs_reward = @showprogress map(Iterators.product(keys(human), implicit_costs)) do ((sigma, alpha, cost), implicit_cost)
    sims = grouped_sims[(sigma, alpha, cost+implicit_cost)]
    payoff, clicks = length(sims) \ mapreduce(+, sims) do t
        [(t.weights' * t.values)[t.choice], length(t.uncovered)]
    end
    net_payoff = payoff - clicks * cost
    (;sigma, alpha, cost, implicit_cost, payoff, clicks, net_payoff)
end
# %% --------
DataFrame(cost_vs_reward[:]) |> CSV.write("results/implicit_cost_summary.csv")




# %% ==================== compute voc (in progress) ====================


# mkpath("tmp/vocs")



# pmap() do (i, cost)
#     vocs, time_ = @timed map(data) do d
#         pol = policies[d.t.sigma, d.t.alpha, cost]
#         b = mutate(d.b, m=pol.m)
#         voc(pol, b)
#     end
#     serialize("$results_path/vocs/$i", (
#         cost=cost,
#         vocs=vocs
#     ))
# end



