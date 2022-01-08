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

# optimize! (this takes 24 hours on a 48 core machine)
mkpath("tmp/policies")
policies = @showprogress pmap(all_mdps; retry_delays=zeros(100)) do m
#policies = @showprogress pmap(all_mdps; retry_delays=zeros(100)) do m
    try
        cache("tmp/policies/" * id(m)) do
            error("Policy not found")  # should already be computed
            optimize_bmps(m; opt_kws..., verbose=false)
        end
    catch err
        @error "Optimization error!" err id(m)
        rethrow()
    end
end;

# %% ==================== simulation ====================

@everywhere function simulate(pol::Policy, s=State(pol.m))::Trial
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

# %% --------
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

# %% =================== cost fitting ====================

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

# %% ==================== simulate human trials ====================

mkpath("results/sims/human_trials")
trials = CSV.read("../data/processed/$version/trials.csv", DataFrame);
filter!(trials) do row
    row.block == "test"
end
unique!(trials, [:cost, :sigma, :alpha, :problem_id])
pol_lookup = Dict(pol.m => pol for pol in policies)

@showprogress pmap(eachrow(trials)) do t
    s = State(Trial(t))
    uncovered = map(1:1000) do i
        simulate(pol_lookup[s.m], s).uncovered
    end
    res = (;t.problem_id, t.sigma, t.alpha, t.cost, t.probabilities, t.payoff_matrix, uncovered)
    write("results/sims/human_trials/$(t.problem_id)-$(t.cost).json", JSON.json(res))
end;

# %% --------

mkpath("results/sims/human_trials_fitcost")

@showprogress pmap(eachrow(trials)) do t
    s = State(Trial(t))
    uncovered = map(1:1000) do i
        m = mutate(s.m, cost=s.m.cost + best_implicit_cost)
        pol_lookup[m]
        simulate(pol_lookup[m], mutate(s; m)).uncovered
    end
    res = (;t.problem_id, t.sigma, t.alpha, t.cost, t.probabilities, t.payoff_matrix, uncovered)
    write("results/sims/human_trials_fitcost/$(t.problem_id)-$(t.cost).json", JSON.json(res))
end;


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

function fit_subset(selector)
    # h = avg_clicks(selector, human)
    argmin(implicit_costs) do implicit_cost
        loss = map(pairs(human)) do ((sigma, alpha, cost), h)
            selector((sigma, alpha, cost)) || return 0
            m = model[(sigma, alpha, cost + implicit_cost)]
            h = human[(sigma, alpha, cost)]
            h - m
        end |> sum |> abs
    end
end
cost_present(cond) = cond[3] > 0


best_by_present = Dict(
    true => fit_subset(cost_present),
    false => fit_subset(!cost_present),
)

get_implicit_cost(cond) = best_by_present[cost_present(cond)]

# %% --------
best_implicit_cost = best_by_cost[1]
mkpath("results/sims/zero_nonzero")
@showprogress for (sigma, alpha, cost) in keys(human)
    sims = grouped_sims[(sigma, alpha, cost + get_implicit_cost((sigma, alpha, cost)))][1:10000]
    sims = map(sims) do t
        mutate(t, cost=cost)
    end
    name = join([sigma, alpha, cost], "-")
    write("results/sims/zero_nonzero/$name.json", JSON.json(sims))
end


# %% ==================== individual fit ====================

trials = group(x->x.pid, all_trials) |> first

ind_sims = @showprogress map(group(x->x.pid, all_trials)) do trials
    t = trials[1]
    best_cost = argmin(implicit_costs) do implicit_cost
        m = model[(t.sigma, t.alpha, t.cost + implicit_cost)]
        h = mean(length(t.uncovered) for t in trials)
        abs(h - m)
    end
    sims = rand(grouped_sims[(t.sigma, t.alpha, t.cost + best_cost)], 200)
    map(sims) do s
        mutate(s; t.cost, trials[1].pid)
    end
end

flatten(ind_sims) |> JSON.json |> write("results/sims/individual.json")

# %% --------

res = @showprogress map(group(x->x.pid, all_trials)) do trials
    t = trials[1]
    implicit_cost = argmin(implicit_costs) do implicit_cost
        m = model[(t.sigma, t.alpha, t.cost + implicit_cost)]
        h = mean(length(t.uncovered) for t in trials)
        abs(h - m)
    end
    sigma, alpha, cost = condition(t)
    (;trials[1].pid, sigma, alpha, cost, implicit_cost)
end 

res |> values |> collect |> DataFrame |> CSV.write("results/implicit_costs.csv")



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



