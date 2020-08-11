module MessagePassing

using GraphicalModels
using GraphicalModels: FGNode, VariableNode, FactorNode, FactorGraph
import Random

"""
    module MessagePassing 

This module contains message-passing algorithms for inference in graphical models.
"""
MessagePassing

abstract type MessagePassingAlgorithm end
# Getter/Setters for messages 
"Get message for edge (`from`,`to`)."
Base.getindex(mp::MessagePassingAlgorithm, from::FGNode, to::FGNode) = Base.getindex(mp.messages, (from,to))
"Set message for edge (`from`,`to`)."
Base.setindex!(mp::MessagePassingAlgorithm, μ, from::FGNode, to::FGNode) = Base.setindex!(mp.messages, μ, (from,to))
"""
    setevidence!(mp::MessagePassingAlgorithm, id::String, value)

Set evidence value on variable identified by `id`.
"""
setevidence!(mp::MessagePassingAlgorithm, id::String, value) = mp.evidence[mp.fg.variables[id]] = value
"Removes evidence from variable identified by `id`."
rmevidence!(mp::MessagePassingAlgorithm, id::String) = delete!(mp.evidence,id)
"""
    reset!(mp::MessagePassingAlgorithm; rndinit=false)

Removes any evidence and resets messages to their initial values (all zero if `rndinit=true`, random otherwise).
"""
function reset!(mp::MessagePassingAlgorithm; rndinit=false) 
    empty!(mp.evidence)
    for μ in values(mp.messages)
        if rndinit
            μ .= log.(rand(length(μ)))
        else
            fill!(μ, 0.0)
        end
    end
    nothing
end

"""
    update!(bp::MessagePassingAlgorithm)

Compute for each edge in factor graph using random flooding scheduling. Returns maximum residual (absolute change in some message).
"""
function update!(bp::MessagePassingAlgorithm)
    # synchronous belief propagation
    # ## compute messages from factor to variable
    # res = 0.0 # residual
    # for f in bp.fg.factors, i in eachindex(f.neighbors)
    #     res = max(res,update!(bp,f,i))
    # end
    # ## compute messages from factors to variables
    # for v in bp.fg.variables, f in v.neighbors
    #     res = max(res,update!(bp,v,f))
    # end
    # Asynchronous belief propagation
    # Assumes graph is connected (will fail otherwise)
    root = Random.rand(collect(values(bp.fg.variables))) # select root variable node at random
    # update messages "away" from this node
    # frontier = FGNode[root] #uncomment for depth-first traversal
    frontier = Set{FGNode}()
    push!(frontier,root)
    visited = Set{FGNode}()
    res = 0.0 # residual
    while !isempty(frontier)
        node = pop!(frontier)
        push!(visited, node)
        if isa(node,VariableNode)
            for f in node.neighbors
                res = max(res,update!(bp,node,f))
                if !(f in visited) && !(f in frontier)
                    push!(frontier, f)
                end
            end
        else
            for (i,v) in enumerate(node.neighbors)
                res = max(res,update!(bp,node,i))
                if !(v in visited) && !(v in frontier)
                    push!(frontier, v)
                end
            end
        end
    end    
    @assert length(visited) == length(bp.fg.variables) + length(bp.fg.factors)            
    bp.iterations += 1
    res
end

# Sum-Product Belief Propagation
include("SumProduct.jl")
# Max-Sum-Product Belief Propagation
include("MixedProduct.jl")

end