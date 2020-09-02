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

function fit_model(model_class)
    pmap(grouped_trials) do (wid, trials)
        wid => fit_model(model_class, trials)
    end |> Dict
end

models = [Optimal, MetaGreedy]
fits = asyncmap(models) do model
    model => fit_model(model)
end |> Dict
serialize("$results_path/fits", fits)
# %% ====================  ====================
trials = grouped_trials[1][2]
data = get_data(trials)
fit_error_model(Optimal(6.), data)



# %% ====================  ====================

pdf = CSV.read("../data/processed/$version/participants.csv");


cost_map = map(eachrow(pdf)) do row
    row.pid => row.cost
end |> Dict

# %% ====================  ====================
g = group(fits[Optimal]) do (wid, fit)
    cost_map[wid]
end;

valmap(g) do fits
    mean([fit[2].model.cost for fit in fits])
end |> sort

# %% ====================  ====================

logps = valmap(fits) do pfits
    length(pfits)
    map(values(pfits)) do fit
        fit.logp
    end
end

logps[Optimal] .- logps[MetaGreedy ]
# %% ====================  ====================
fits[MetaGreedy]

# %% ====================  ====================
valmap(group(x->x.cost, all_trials)) do trials
    map(trials) do t
        length(t.uncovered)
    end |> mean
end |> sort



