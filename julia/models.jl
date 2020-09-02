MetaMDP(m::MetaMDP; cost) = MetaMDP(
    m.n_gamble,
    m.n_outcome,
    m.reward_dist,
    m.weight_dist,
    cost,
)

# %% ==================== MetaGreedy ====================

struct MetaGreedy <: Model
    cost::Float64
end

instantiations(::Type{MetaGreedy}) = map(MetaGreedy, 0:2:50)

function preferences(model::MetaGreedy, b::Belief)
    m = MetaMDP(b.m; cost=model.cost)
    b = Belief(m, b)
    voc1(b)
end


# # %% ==================== Optimal ====================
struct Optimal <: Model
    cost::Float64
end

instantiations(::Type{Optimal}) = map(Optimal, costs)

function load_qs()
    data = load_trials(version) |> get_data;
    if !isdir("$results_path/vocs")
        @warn "Couldn't find $results_path/vocs"
        return nothing
    end
    # check = checksum(data)
    all_qs = map(readdir("$results_path/vocs")) do i
        qq = deserialize("$results_path/vocs/$i")
        # @assert qq.checksum == check
        @assert length(data) == length(qq.vocs)
        qdict = map(data, qq.vocs) do d, q
            hash(d) => [0; q]
        end |> Dict
        qq.cost => qdict
    end |> Dict
end

new_precomputed_qs = load_qs();

function preferences(model::Optimal, d::Datum)
    new_precomputed_qs[model.cost][hash(d)]
end


