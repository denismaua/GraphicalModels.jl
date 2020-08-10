using GraphicalModels
import GraphicalModels: FactorGraph, FGNode
import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, marginal, decode, setevidence!, setmapvar!

# load model form file
# fg = FactorGraph("/Users/denis/code/GraphicalModels/test/example.uai")
fg = FactorGraph("/Users/denis/learned-spns/spambase/spambase.uai")
evid = Dict( "4" => 2, "7" => 2 )
mapvars = ["0", "3"]
# initialize messages 
bp = HybridBeliefPropagation(fg; rndinit=true)
# bp.normalize = true
bp.λ = 0.9 # dampening factor to improve convergence
# set map variables
setevidence!("0", 1)
setevidence!("3", 1)
# setmapvar!(bp, "0")
# setmapvar!(bp, "1")
# Set evidence
for (var,value) in evid
    setevidence!(bp, var, value)
end
# Find scheduling
root = fg.variables[string(length(fg.variables)-1)]
frontier = Vector{FGNode}()
push!(frontier,root)
visited = Set{FGNode}()
scheduling = Vector{FGNode}()
res = 0.0 # residual
while !isempty(frontier)
    node = popfirst!(frontier)
    push!(visited, node)
    push!(scheduling, node)
    for n in node.neighbors
        if !(n in visited) && !(n in frontier)
            push!(frontier, n)
        end
    end
end
# now run algorithm
for i=1:10
    res = 0.0
    for node in scheduling
        for ne in node.neighbors
            res = max(res, update!(bp, node, ne))
        end
    end
    for node in Iterators.reverse(scheduling)
        for ne in node.neighbors
            res = max(res, update!(bp, node, ne))
        end
    end
    println("$i \t $res \t $(decode(bp, "0")) \t $(decode(bp, "1"))")
end
# @show marginal(bp, "0") 
# @show marginal(bp, "1") 
# @show marginal(bp, "16") 

# for i=1:10
#     @info i
#     @show update!(bp)
#     @show marginal(bp, "0") decode(bp, "0")
#     @show marginal(bp, "1") decode(bp, "1")
# end

# decode
println()
println("X0 = $(decode(bp, "0"))")
println("X1 = $(decode(bp, "1"))");
