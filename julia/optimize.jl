using Serialization
import CSV
using Distributed

@everywhere begin
    include("meta_mdp.jl")
    include("bmps.jl")
    include("optimize_bmps.jl")
    include("params.jl")
    include("utils.jl")
end
# include("results.jl")
# include("data.jl")
# include("models_base.jl")
# include("models.jl")

save_path = "$results_path/mdps"

function write_sbatch(n)
    script = """
    #!/usr/bin/env bash
    #SBATCH --job-name=optimize
    #SBATCH --output=out/slurm/%A_%a
    #SBATCH --array=1-$n
    #SBATCH --time=75
    #SBATCH --mem-per-cpu=500
    #SBATCH --mail-type=end
    #SBATCH --mail-user=flc2@princeton.edu

    module load julia
    julia optimize.jl \$SLURM_ARRAY_TASK_ID
    """
    open("optimize.sbatch", "w") do f
        write(f, script)
    end
    println("Wrote optimize.sbatch")
end

if ARGS[1] == "prepare"
    pdf = CSV.read("../data/processed/$version/participants.csv");
    conditions = pdf[[:alpha, :sigma]] |> unique |> eachrow |> map(copy)
    all_mdps = map(Iterators.product(conditions, costs)) do (cond, cost)
        MetaMDP(
            reward_dist=Normal(0, cond.sigma),
            weight_dist=Dirichlet(ones(4) .* cond.alpha),
            cost=cost
        )
    end |> collect

    for (i, m) in enumerate(all_mdps)
        mkpath("$save_path/$i")
        serialize("$save_path/$i/mdp", m)
    end
    println("Saved MDPs to $save_path")
    write_sbatch(length(all_mdps))
else
    i = parse(Int, ARGS[1])
    m = deserialize("$save_path/$i/mdp")
    println(m)

    kws = (seed=1, n_iter=500, n_roll=10000, α=100)
    @time pol = optimize_bmps(m; kws..., verbose=true)
    serialize("$save_path/$i/policy", (
        policy=pol,
        kws=kws
    ))
end