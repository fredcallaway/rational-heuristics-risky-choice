# %% ==================== Set up ====================

using StatsBase
using DataStructures: OrderedDict
using Lazy: @as

include("utils.jl")
include("meta_mdp.jl")
include("data.jl")

version = "1.0"
all_trials = load_trials(version);
mkpath("results")

# %% ==================== Define strategies ====================

C = reshape(1:24, 4, 6)  # matrix of cell numbers
STRATEGIES = [:rand, :wadd, :ttb, :sat_ttb, :unknown]

function classify(t::Trial)
    # no clicking => rand
    length(t.uncovered) == 0 && return :rand
    # full clicking => wadd
    length(t.uncovered) == length(C) && return :wadd
    
    # click whole maximal weight row => ttb
    ttb_row = argmax(t.weights)
    Set(t.uncovered) == Set(C[ttb_row, :]) && return :ttb
    
    # click subest of maximal weight row AND last revealed is unique maximum => sat_ttb
    Set(t.uncovered) < Set(C[ttb_row, :]) && 
        argmax(t.values[t.uncovered]) == length(t.uncovered) && return :sat_ttb

    return :unknown
end

# %% ==================== Descriptive statistics ====================

term_reward(t::Trial) = term_reward(get_data(t)[end].b)

# %% ==================== Classify and apply descriptive statistics  ====================

# one row for each trial
T = map(all_trials) do t
    (
        sigma = t.sigma, alpha=t.alpha, cost=t.cost, t=t,
        strategy = classify(t), 
        n_revealed = length(t.uncovered),
        term_reward = term_reward(t),
    ) 
end |> DataFrame

combine(groupby(T, [:sigma, :cost]), :n_revealed => mean) |> sort


# %% ==================== Strategy frequencies ====================

ivs = [:sigma, :alpha, :cost]
# one row for each condition
S = @as x T.strategy begin
    x .== reshape(STRATEGIES, 1, :)
    DataFrame(x, STRATEGIES)
    hcat(T[ivs], x)
    groupby(x, ivs)
    combine(x, STRATEGIES .=> sum)
    rename(col->replace(col, "_sum" => ""), x)
    sort!
end
S[4:end] = S[4:end] ./ sum.(eachrow(S[4:end]))
CSV.write("results/strategy_frequencies.csv", S)


# %% ==================== Breakdowns ====================

combine(S, STRATEGIES .=> mean)
combine(groupby(S, :sigma), STRATEGIES .=> mean)
combine(groupby(S, :alpha), STRATEGIES .=> mean)
combine(groupby(S, :cost), STRATEGIES .=> mean)
