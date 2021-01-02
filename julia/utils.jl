using SplitApplyCombine
using Serialization
flatten = SplitApplyCombine.flatten

function cache(f, file)
    isfile(file) && return deserialize(file)
    result = f()
    serialize(file, result)
    result
end

# ---------- Printing stuff ---------- #

using Printf
function describe_vec(x::Vector)
    @printf("%.3f ± %.3f  [%.3f, %.3f]\n", juxt(mean, std, minimum, maximum)(x)...)
end
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

function writev(fn)
    x -> begin
        write(fn, x)
        run(`du -h $fn`)
    end
end
Base.write(fn) = x -> write(fn, x)

# ---------- Math and matrix stuff ---------- #
nanreduce(f, x) = f(filter(!isnan, x))
nanmean(x) = nanreduce(mean, x)
nanstd(x) = nanreduce(std, x)
Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
Base.reshape(idx::Union{Int,Colon}...) = x -> reshape(x, idx...)

# ---------- Type stuff ---------- #
dictkeys(d::Dict) = (collect(keys(d))...,)
dictvalues(d::Dict) = (collect(values(d))...,)

type2nt(p) = (;(v=>getfield(p, v) for v in fieldnames(typeof(p)))...)
fields(p) = [getfield(p, v) for v in fieldnames(typeof(p))]

namedtuple(d::Dict{String,T}) where {T} =
    NamedTuple{Symbol.(dictkeys(d))}(dictvalues(d))

namedtuple(d::Dict{Symbol,T}) where {T} =
    NamedTuple{dictkeys(d)}(dictvalues(d))

function mutate(x::T; kws...) where T
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

getfields(x) = (getfield(x, f) for f in fieldnames(typeof(x)))

# ---------- Functional stuff ---------- #
Base.map(f::Function) = xs -> map(f, xs)
Base.map(f::Type) = xs -> map(f, xs)
Base.map(f, d::AbstractDict) = [f(k, v) for (k, v) in d]
valmap(f, d::AbstractDict) = Dict(k => f(v) for (k, v) in d)
valmap(f) = d->valmap(f, d)
keymap(f, d::AbstractDict) = Dict(f(k) => v for (k, v) in d)
juxt(fs...) = x -> Tuple(f(x) for f in fs)
repeatedly(f, n) = [f() for i in 1:n]


function Base.get(collection, key)
    v = get(collection, key, "__missing__")
    if v == "__missing__"
        error("Key $key not found in collection!")
    end
    v
end

Base.get(key) = (collection) -> get(collection, key)

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

macro ifundef(exp)
    local e = :($exp)
    isdefined(Main, e.args[1]) ? :($(e.args[1])) : :($(esc(exp)))
end

# ---------- Hashing stuff ---------- #

function hash_struct(s, h::UInt64=UInt64(0))
    reduce(getfields(s); init=h) do acc, x
        hash(x, acc)
    end
end

function struct_equal(s1::T, s2::T) where T
    all(getfield(s1, f) == getfield(s2, f)
        for f in fieldnames(T))
end
