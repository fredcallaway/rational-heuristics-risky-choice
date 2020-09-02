using Distributed
@everywhere begin
    include("params.jl")
    include("utils.jl")
    include("meta_mdp.jl")
    include("bmps.jl")
    include("data.jl")
    include("models_base.jl")
    include("models.jl")
end

# %% ====================  ====================

version = "1.0"
all_trials = load_trials(version);
grouped_trials = group(x->x.pid, all_trials) |> collect;

function make_viz(t::Trial)
    n_outcome, n_gamble = size(t.values)  
    uninvert_index = transpose(reshape(1:(n_outcome * n_gamble), n_gamble, n_outcome))
    (
        payoff_matrix = t.values |> transpose |> splitdims .|> map(Int),
        probabilities = t.weights,
        cost = t.cost,
        demo = (
            clicks = uninvert_index[t.uncovered] .- 1,
            choice = t.choice - 1
        )
    )
end

function simulate(t::Trial, cost)
    m = MetaMDP(t)
    uncovered = Int[]
    roll = rollout(MetaGreedyPolicy(1e4, m), State(t)) do b, c
        c != ⊥ && push!(uncovered, c)
    end
    choice = sample(Weights(choice_probs(roll.belief)))
    mutate(t, uncovered=uncovered, choice=choice)
end

# %% ====================  ====================
out = "/Users/fred/heroku/mouselab/static/json/viz"
g = group(grouped_trials) do gt
    (gt[1].cost, gt[1].sigma)
end
trials = map(x->flatten(x[1:5]), g) |> flatten

map(collect(group(x->x.pid, trials))) do trials
    t = trials[1]
    make_viz.(trials) |> JSON.json |> write("$out/$(t.pid).json")
    (cost=t.cost, sigma=t.sigma, pid=t.pid)
end |> sort |> JSON.json |> writev("$out/table.json")

sort(table)


# %% --------

map(grouped_trials) do gt
    gt[1].cost, gt[1].sigma
end





# %% --------

trials = map(grouped_trials) do tt
    tt[20]
end
trials = trials[randperm(length(trials))[1:100]]
trials .|> make_viz |> JSON.json |> writev("/Users/fred/heroku/mouselab/static/json/viz/test.json")

map(trials) do t
    simulate(t, )
end |> make_viz |> JSON.json |> writev("/Users/fred/heroku/mouselab/static/json/viz/test.json")
# %% --------
sims = trials[1:10] .|> simulate
trials[2].uncovered

sims .|> make_viz |> JSON.json |> writev("/Users/fred/heroku/mouselab/static/json/viz/test.json")

# %% --------
row = df[10, :]
t = Trial(row)
v = make_viz(t)
@assert JSON.parse(row.payoff_matrix) == v.payoff_matrix
@assert row.choice_index == v.demo.choice
@assert JSON.parse(row.clicks) == v.demo.clicks



# %% ==================== OLD ====================
sim_data = map(all_trials) do t
    simulate(t)
end |> flatten;

fit_model(MetaGreedy, sim_data)
fit_error_model(MetaGreedy(4.), sim_data)
map(0:2:40) do c
    fit_error_model(MetaGreedy(c), sim_data).logp
end

model = MetaGreedy(5.)
preferences(model, sim_data[1].b)
