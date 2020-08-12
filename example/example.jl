# Loads factor graph, query and evidence from file in UAI Competition format
# and runs hybrid message passing to compute marginal MAP inference.

using GraphicalModels
import GraphicalModels: FactorGraph, FGNode
import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, marginal, decode, setevidence!, setmapvar!

# Filenames
model_filename = "$(@__DIR__)/nltcs.uai"
evidence_filename = "$(@__DIR__)/nltcs.evid"
query_filename = "$(@__DIR__)/nltcs.query"

# spambase
# model_filename = "/Users/denis/learned-spns/spambase/spambase.uai"
# evidence_filename = "/Users/denis/learned-spns/spambase/spambase.evid"
# query_filename = "/Users/denis/learned-spns/spambase/spambase.query"

# ionosphere
# model_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.uai"
# evidence_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.evid"
# query_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.query"

# Mushrooms
# model_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.uai"
# evidence_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.evid"
# query_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.query"

# nips
# model_filename = "/Users/denis/learned-spns/nips/nips.uai"
# evidence_filename = "/Users/denis/learned-spns/nips/nips.evid"
# query_filename = "/Users/denis/learned-spns/nips/nips.query"


# load model from file
@time fg = FactorGraph(model_filename)
println("Loaded: $(length(fg.variables)) variables, $(length(fg.factors)) factors.")
# initialize messages 
bp = HybridBeliefPropagation(fg; rndinit=true) # set rndinit=true for noninformative initalization (messages = constant)
# bp.normalize = true # uncomment to normalize messages after update (makes their sum in linear domain = 1; might improve convergence)
# load evidence from file
open(evidence_filename) do io
    line = readline(io) # read line
    fields = split(line) # split into fields
    no_evid = parse(Int, fields[1]) # first field is no. of evidence variables
    for i = 2:2:length(fields)
        setevidence!(bp, string(fields[i]), parse(Int, fields[i+1])+1) # add one to value since julia uses 1-based indexing (and uai format uses 0-based indexing)
    end
    @assert length(bp.evidence) == no_evid
    println("Loaded: $no_evid evidence variables.")
end
# load evidence from file
open(query_filename) do io
    line = readline(io) # read line
    fields = split(line) # split into fields
    no_query = parse(Int, fields[1]) # first field is no. of evidence variables
    for i = 2:length(fields)
        setmapvar!(bp, string(fields[i])) 
    end
    @assert length(bp.mapvars) == no_query
    println("Loaded: $no_query query variables.")
end
# Find scheduling
print("building message scheduling...")
etime =  @elapsed begin
    root = fg.variables[string(length(fg.variables)-1)]
    frontier = Vector{FGNode}()
    push!(frontier,root)
    visited = Set{FGNode}()
    scheduling = Vector{Tuple{FGNode,FGNode}}()
    res = 0.0 # residual
    while !isempty(frontier)
        node = popfirst!(frontier)
        push!(visited, node)
        for n in node.neighbors
            push!(scheduling, (node,n)) # add outgoing messages
            if !(n in visited) && !(n in frontier)
                push!(frontier, n)
            end
        end
    end
end
println("done [$(etime)s].")
# now run algorithm
for i=1:10
    # bp.λ = 0.99^(i-1) # uncomment to apply exponential decay to updates (improves convergence)
    # etime = @elapsed res = update!(bp)
    etime = @elapsed begin
        res = 0.0 # residual
        if i % 2 == 1
            for (from,to) in scheduling
                res = max(res, update!(bp, from, to))
            end
        else
            for (from,to) in Iterators.reverse(scheduling)
                res = max(res, update!(bp, from, to))
            end
        end
    end
    if length(bp.mapvars) > 100
        println("[$i] \t $(etime)s \t $res")
    else
        println("[$i] \t $(etime)s \t $res \t ", join(map(v -> decode(bp,v), collect(bp.mapvars)), " "))
    end
    if res < 1e-10 break end # early stop
end
# decode
print(length(bp.mapvars))
for v in bp.mapvars
    i = findfirst(isequal(v),fg.variables)
    print(" $i $((decode(bp, v))-1)")
end
println()
