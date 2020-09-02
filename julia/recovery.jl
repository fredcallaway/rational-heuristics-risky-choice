

function simulate(t::Trial)
    m = MetaMDP(t)
    data = Datum[]
    rollout(MetaGreedyPolicy(1., m)) do b, c
        push!(data, Datum(t, copy(b), c))
    end
    data
end


# %% ====================  ====================
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




# map(all_trials[1:2000]) do t
#     simulate(t)