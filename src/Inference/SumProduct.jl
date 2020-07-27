# Message Passing for Marginal and LogPartition Computation
 
abstract type MessagePassingAlgorithm end
"Data structure for sum-product belief propagation algorithm."
mutable struct BeliefPropagation <: MessagePassingAlgorithm
    fg::FactorGraph
    evidence::Dict{Int,UInt}
    iterations::Int
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    "Initialize belief propagation with no evidence."
    function BeliefPropagation(fg::FactorGraph) 
        # initialize messages as vectos of ones
        μ = Dict{Tuple{FGNode,FGNode},AbstractVector}()
        for v in fg.variables, f in v.neighbors
                μ[v,f] = ones(length(v.variable))
        end
        for f in fg.factors, v in f.neighbors
                μ[f,v] = ones(length(v.variable))
        end        
        new(fg, Dict{Variable,UInt}(), 0, μ)
    end
end

"Update belief propagation message from variable to factor node."
function update!(bp::BeliefPropagation, from::VariableNode, to::FactorNode)
    fill!(bp.messages[from,to],1.0)
    for factor in from.neighbors
        if factor ≠ to
            bp.messages[from,to] .*= bp.messages[factor,from]
        end
    end    
    bp.messages[from,to]
end

"Update belief propagation message from factor to variable node."
function update!(bp::BeliefPropagation, from::FactorNode, to::VariableNode)
    fill!(bp.messages[from,to],1.0)
    for variable in from.neighbors
        if variable ≠ to
            #TODO!
            bp.messages[from,to] .*= bp.messages[factor,from]
        end
    end    
    bp.messages[from,to]
end

"Computes belief propagation messages for each edge in factor graph."
function update!(bp::BeliefPropagation)
    for v in bp.fg.variables, f in v.neighbors
        update!(bp,v,f)
    end
# for ((from,to),μ) in bp.messages
    #     println(typeof(from), "->", typeof(to), ": ", μ)
    # end    
    nothing
end

# extract all marginals from belief propagation
function allmarginals(mp::BeliefPropagation)
    1.0
end
