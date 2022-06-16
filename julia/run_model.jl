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

# version = "1.0"
version = "2.3"
implicit_costs = [0:0.1:3; 4:17]

participants = CSV.read("../data/human/$version/participants.csv", DataFrame);

alphas = participants.alpha |>  unique |> sort
sigmas = participants.sigma |> unique |> sort
explicit_costs = participants.cost |> unique |> sort
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
all_sims = @showprogress asyncmap(policies) do pol
    deserialize("tmp/sims/" * id(pol.m))
end;

open("../data/model/all_sims.json", "w") do f
    JSON.print(f, all_sims)
end

# %% ==================== cost fitting ====================

dictionary = SplitApplyCombine.dictionary
condition(t::Trial) = map(float, (t.sigma, t.alpha, t.cost))
condition(m::MetaMDP) = (m.reward_dist.σ, m.weight_dist.alpha[1], m.cost)

grouped_sims = map(all_mdps, all_sims) do m, trials
    condition(m) => trials
end |> dictionary

nclick_model = map(grouped_sims) do trials
    mean(length(t.uncovered) for t in trials)
end

function fit_implicit_cost(trial_data)
    all_trials = map(Trial, eachrow(trial_data))
    nclick_human = map(group(condition, all_trials)) do trials
        mean(length(t.uncovered) for t in trials)
    end

    argmin(implicit_costs) do implicit_cost
        map(pairs(nclick_human)) do ((sigma, alpha, cost), h)
            m = nclick_model[(sigma, alpha, cost + implicit_cost)]
            # (h - m) ^ 2
            # abs(h - m)
            h - m
        end |> sum |> abs
    end
end

# %% ==================== simulate human trials ====================

using DataFramesMeta
trial_data = @chain "../data/human/$version/trials.csv" begin
    CSV.read(DataFrame)
    @rsubset :block == "test"
    @rtransform :n_click = Int(:click_cost / :cost)
end
trial_data = @chain trial_data begin
    groupby(:pid)
    @combine :is_random = mean(:n_click .== 0) > 0.5
    rightjoin(trial_data, on=:pid)
end

# %% --------

function write_simulation(path, trials; fit_cost=false, repeats=1000)
    mkpath(path)
    pol_lookup = Dict(pol.m => pol for pol in policies)
    to_sim = eachrow(unique(trials, [:cost, :sigma, :alpha, :problem_id]))
    
    implicit_cost = fit_cost ? fit_implicit_cost(trials) : 0.
    @info "Simulating" implicit_cost path nrow(trials)
    @showprogress pmap(to_sim) do t
        s = State(Trial(t))
        m = mutate(s.m, cost=s.m.cost + implicit_cost)
        pol = pol_lookup[m]
        uncovered = map(1:repeats) do i
            simulate(pol, mutate(s; m)).uncovered
        end
        res = (;t.problem_id, t.sigma, t.alpha, t.cost, t.probabilities, t.payoff_matrix, uncovered)
        write("$path/$(t.problem_id)-$(t.cost).json", JSON.json(res))
    end
end

for (key, td) in pairs(groupby(trial_data, :display_ev))
    cond = key.display_ev ? "exp" : "con"
    base = "../data/model/human_trials_exp2_$cond"

    write_simulation(base, td)
    write_simulation("$(base)_fitcost", td; fit_cost=true)

    excl_td = @rsubset(td, !:is_random)
    write_simulation("$(base)_fitcost_exclude", excl_td; fit_cost=true)
    write_simulation("$(base)_exclude", excl_td; fit_cost=false)
end
