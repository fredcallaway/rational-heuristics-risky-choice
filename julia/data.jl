using JSON
using CSV
using DataFrames
using SplitApplyCombine
using Memoize

parse_matrix(m) = m |> JSON.parse .|> Vector{Float64} |> combinedims |> transpose |> collect

struct Trial
    pid::Int
    trial_index::Int
    sigma::Int
    alpha::Float64
    cost::Int
    # problem_id::Int
    weights::Vector{Float64}
    values::Matrix{Float64}
    uncovered::Vector{Int}
    choice::Int
end

Base.show(io::IO, t::Trial) = print(io, "T($(t.pid), $(t.trial_index))")
Base.hash(t::Trial, h::UInt64) = hash((t.pid, t.trial_index), h)

function Trial(row::DataFrameRow)
    # converts indices from row-major to column-major
    n_outcome, n_gamble = size(parse_matrix(row.payoff_matrix))
    invert_index = transpose(reshape(1:(n_outcome * n_gamble), n_outcome, n_gamble))

    Trial(
        row.pid,
        row.trial_index + 1,
        row.sigma,
        row.alpha,
        row.cost, 
        float.(JSON.parse(row.probabilities)),
        parse_matrix(row.payoff_matrix),
        invert_index[Int.(JSON.parse(row.clicks)) .+ 1],
        row.choice_index + 1,
    )
end

@memoize function load_trials(version)
    df = DataFrame(CSV.File("../data/human/$version/trials.csv"));
    filter!(df) do row
        row.block == "test"
    end
    map(Trial, eachrow(df))
end

struct Datum
    t::Trial
    b::Belief
    c::Int
    # c_last::Union{Int, Nothing}
end
Base.hash(d::Datum, h::UInt64) = hash(d.c, hash(d.t, h))

function MetaMDP(t::Trial, cost::Real)
    no, ng = size(t.values)
    MetaMDP(ng, no,
        Normal(0, t.sigma),
        Dirichlet(ones(no) * t.alpha),
        cost
    )
end
MetaMDP(t::Trial) = MetaMDP(t, t.cost)
State(m::MetaMDP, t::Trial) = State(m, t.weights .* t.values, t.weights)
State(t::Trial) = State(MetaMDP(t), t.weights .* t.values, t.weights)

function get_data(t::Trial)
    m = MetaMDP(t)
    s = State(m, t)
    b = Belief(s)
    data = Datum[]

    for c in t.uncovered
        push!(data, Datum(t, copy(b), c))
        observe!(b, s, c)
    end
    push!(data, Datum(t, b, ⊥))
    data
end

get_data(trials::Vector{Trial}) = flatten(map(get_data, trials))
