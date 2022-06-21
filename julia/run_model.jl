using Serialization
using CSV
using Distributed
using ProgressMeter
using DataFramesMeta


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

version = isempty(ARGS) ? "1.0" : ARGS[1]
version = "2.3"
implicit_costs = [0:0.1:3; 4:17]
experiment = startswith(version, "2") ? 2 : 1
if experiment == 2
    implicit_costs = -3:0.1:3
end
@info "Running model" version implicit_costs

participants = CSV.read("../data/human/$version/participants.csv", DataFrame);

alphas = participants.alpha |>  unique |> sort
sigmas = participants.sigma |> unique |> sort
explicit_costs = participants.cost |> unique |> sort

costs = @chain Iterators.product(explicit_costs, implicit_costs) begin
    sum.()
    round.(digits=1)
    unique
end

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
policies = @showprogress "Policies " pmap(all_mdps; retry_delays=zeros(3)) do m
    try
        cache("tmp/policies/" * id(m)) do
            error("Policy not found")  # should already be computed
            optimize_bmps(m; opt_kws..., verbose=false)
        end
    catch err
        @error "Optimization error!" err id(m)
        rethrow()
    end
end

pol_lookup = Dict(id(pol.m) => pol for pol in policies)

@everywhere pol_lookup = $pol_lookup


# %% ==================== load data ====================

trial_data = @chain "../data/human/$version/trials.csv" begin
    CSV.read(DataFrame)
    @rsubset :block == "test"
    @rtransform :n_click = length(JSON.parse(:clicks))
    # @rtransform :n_click = Int(:click_cost / :cost)
end

trial_data = @chain trial_data begin
    groupby(:pid)
    @combine :is_random = mean(:n_click .== 0) > 0.5
    rightjoin(trial_data, on=:pid)
end

# %% ==================== simulation ====================

@everywhere function simulate(pol::Policy, s::State)
    uncovered = Int[]
    rollout(pol, s) do b, c
        c != ⊥ && push!(uncovered, c)
    end
    uncovered
end

@everywhere function simulate(s::State; implicit_cost=0, N=1)
    m = mutate(s.m, cost=round(s.m.cost + implicit_cost; digits=1))
    s = mutate(s; m)
    pol = pol_lookup[id(m)]
    repeatedly(N) do
        simulate(pol, s)
    end
end

# %% ==================== cost fitting ====================

unique_trials(trial_data) = eachrow(unique(trial_data, [:sigma, :cost, :problem_id]))

function compute_model_nclick(trial_data)
    jobs = Iterators.product(unique_trials(trial_data), implicit_costs)
    @showprogress "compute nclick " pmap(jobs) do (trial, implicit_cost)
        k = trial.sigma, trial.cost, trial.problem_id, implicit_cost
        s = State(Trial(trial))
        n_click = mapreduce(length, +, simulate(s; implicit_cost, N=1000)) / 1000
        k => n_click
    end
end

nclick_data = cache("tmp/nclick-$version") do
    compute_model_nclick(trial_data)
end
model_nclick = Dict(nclick_data[:])

function fit_implicit_cost(trial_data)
    human = mean(trial_data.n_click)
    argmin(implicit_costs) do implicit_cost
        model = mean(eachrow(trial_data)) do t
            model_nclick[t.sigma, t.cost, t.problem_id, implicit_cost]
        end
        abs(model - human)
    end
end

# %% ==================== simulate human trials ====================

function write_simulation(path, trials; fit_cost=false, N=1000)    
    implicit_cost = fit_cost ? fit_implicit_cost(trials) : 0.
    println("$path: $implicit_cost")
    mkpath(path)

    trial_counts = @chain trial_data begin
        groupby([:sigma, :cost, :problem_id])
        combine(nrow)
        @rtransform :key = (:sigma, :cost, :problem_id)
        select([:key, :nrow])
        eachrow
        Dict
    end

    @showprogress "Simulating " pmap(unique_trials(trials)) do t
        s = State(Trial(t))
        uncovered = simulate(s; implicit_cost, N)
        n_human = trial_counts[(t.sigma, t.cost, t.problem_id)]
        res = (;t.problem_id, t.sigma, t.alpha, t.cost, t.probabilities, t.payoff_matrix, 
                uncovered, n_human)
        write("$path/$(t.problem_id)-$(t.cost)-$(t.sigma).json", JSON.json(res))
    end
    nothing
end

if experiment == 1
    base = "../data/model/exp1"
    write_simulation(base, trial_data)
    write_simulation("$(base)_fitcost", trial_data; fit_cost=true)
    excl_trial_data = @rsubset(trial_data, !:is_random)
    write_simulation("$(base)_fitcost_exclude", excl_trial_data; fit_cost=true)

else @assert experiment == 2
    base = "../data/model/exp2"
    write_simulation(base, trial_data)

    for (key, td) in pairs(groupby(trial_data, :display_ev))
        cond = key.display_ev ? "exp" : "con"
        write_simulation("$(base)_$(cond)_fitcost", td; fit_cost=true)
        excl_td = @rsubset(td, !:is_random)
        write_simulation("$(base)_$(cond)_fitcost_exclude", excl_td; fit_cost=true)
    end
end
