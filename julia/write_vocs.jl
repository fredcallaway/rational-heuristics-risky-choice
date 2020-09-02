using Distributed
@everywhere begin
    using Serialization
    include("meta_mdp.jl")
    include("bmps.jl")
    include("data.jl")
    include("utils.jl")
    include("params.jl")
end

# job = 1
# job = parse(Int, ARGS[1])


# %% ====================  ====================

function write_vocs(policies, data)
    mkpath("$results_path/vocs")
    pmap(enumerate(costs)) do (i, cost)
        vocs, time_ = @timed map(data) do d
            pol = policies[d.t.sigma, d.t.alpha, cost]
            b = mutate(d.b, m=pol.m)
            voc(pol, b)
        end
        serialize("$results_path/vocs/$i", (
            cost=cost,
            vocs=vocs
        ))
    end
end

all_trials = load_trials(version);
all_data = get_data(all_trials);
policies = asyncmap(1:260) do i
    pol = deserialize("$results_path/mdps/$i/policy").policy
    σ = pol.m.reward_dist.σ
    α = pol.m.weight_dist.alpha[1]
    cost = pol.m.cost
    (σ, α, cost) => pol
end |> Dict;

@time write_vocs(policies, all_data)
