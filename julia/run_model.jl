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

@everywhere id(m::MetaMDP) = join([round(m.weight_dist.alpha[1]; digits=3), m.reward_dist.σ, m.cost], "-")

# %% ==================== setup ====================
# push!(ARGS, "1", "TEST")
EXPERIMENT = parse(Int, ARGS[1])
RUN = ARGS[2]

version = ["1.0", "2.3"][EXPERIMENT]
implicit_costs = 0:0.1:3

@info "Running model" EXPERIMENT RUN

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
@everywhere opt_kws = (seed=1, n_iter=500, n_roll=10000, α=100, parallel=false)

# optimize! (this takes 24 hours on a 48 core machine)
mkpath("tmp/$RUN/policies")
policies = @showprogress "Policies " pmap(all_mdps; retry_delays=zeros(3)) do m
    try
        cache("tmp/$RUN/policies/" * id(m)) do
            # error("Policy not found")  # should already be computed
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
end

trial_data = @chain trial_data begin
    groupby(:pid)
    @combine :is_random = mean(:n_click .== 0) > 0.5
    rightjoin(trial_data, on=:pid)
end

if EXPERIMENT == 2
    bad_pids = @chain "
        1  17  30  33  35  60  79 109 110 119 127 132 171
        176 186 188 197 199 202 208 212 228 236 250 252 263
        265 266 288 298 300 315 348 353 356 374 381 395 417
        425 482 485 510 516 526 529 563 567 569 580 587 589
        590 606
        2   4   5  26  57  69  86  92  96  97 104 105 114
        136 145 155 158 165 190 231 235 241 242 245 247 283
        308 313 318 324 327 347 375 382 387 396 406 412 413
        442 455 478 499 507 527 528 536 537 560 570 574 583
        584 595 599 610
    " strip split(r"[\n ]+") parse.(Int, _)

    trial_data = @rtransform(trial_data, :low_performance = :pid in bad_pids)
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
        n_click = mapreduce(length, +, simulate(s; implicit_cost, N=10_000)) / 10_000
        k => n_click
    end
end

nclick_data = cache("tmp/$RUN/nclick-$version") do
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
        write("$path/$(t.problem_id)-$(t.cost)-$(t.sigma).json", json(res))
    end
    implicit_cost
end


if EXPERIMENT == 1
    base = "../data/model/$RUN/exp1"
    write_simulation(base, trial_data)
    full = write_simulation("$(base)_fitcost", trial_data; fit_cost=true)
    excl_trial_data = @rsubset(trial_data, !:is_random)
    exclude = write_simulation("$(base)_fitcost_exclude", excl_trial_data; fit_cost=true)
    @chain (;full, exclude) json write("../data/model/$RUN/exp1_costs.json", _)
else @assert EXPERIMENT == 2
    base = "../data/model/$RUN/exp2"
    write_simulation(base, trial_data)

    costs = map(pairs(groupby(trial_data, :display_ev))) do (key, td)
        cond = key.display_ev ? "exp" : "con"
        full = write_simulation("$(base)_$(cond)_fitcost", td; fit_cost=true)
        
        excl_td = @rsubset(td, !:is_random)
        exclude = write_simulation("$(base)_$(cond)_fitcost_exclude", excl_td; fit_cost=true)
        
        excl_td = @rsubset(td, !:low_performance)
        exclude_alt = write_simulation("$(base)_$(cond)_fitcost_exclude_alt", excl_td; fit_cost=true)

        cond => (;full, exclude, exclude_alt)
    end
    @chain costs Dict json write("../data/model/$RUN/exp2_costs.json", _)
end

# %% ==================== compute perfect rationality benchmark ====================

if EXPERIMENT == 1
    @everywhere function emax_normal(k::Real, d::Normal)
        mcdf(x) = cdf(d, x)^k
        lo = d.μ - 5d.σ; hi = d.μ + 5d.σ
        - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
    end

    baselines = map(alphas) do alpha
        m = MetaMDP(reward_dist=Normal(0, 1), weight_dist=Dirichlet(ones(4) .* alpha), n_gamble=1)
        n_total = 1_000_000
        n_batch = 1000
        n_per = n_total / n_batch
        x = @showprogress pmap(1:n_batch) do i
            mean(1:n_per) do j
                b = Belief(m)
                gv = gamble_values(b)[1]
                emax_normal(6, gv)
            end
        end
        (;mean=mean(x), sem = std(x) / √n_batch)
    end

    df = mapreduce(vcat, sigmas) do sigma
        µ, sem = invert(baselines)
        DataFrame(;sigma, alpha=alphas, baseline = µ .* sigma, sem = sem .* sigma)
    end

    df |> CSV.write("../data/model/$RUN/perfectly_rational.csv")
end
