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
    μ = bp.messages[from,to]
    fill!(μ,1.0)
    # compute product of incoming messages
    for factor in from.neighbors
        if factor ≠ to
            μ .*= bp.messages[factor,from]
        end
    end    
    μ
end

"Update belief propagation message from factor to its ith neighbor."
function update!(bp::BeliefPropagation, from::FactorNode, to::Integer)
# function update!(bp::BeliefPropagation, from::FactorNode, to::VariableNode)
    # fill!(bp.messages[from,to],0.0)
    μ = bp.messages[from,from.neighbors[to]]
    fill!(μ,0.0)
    # collect incoming messages
    μ_in = [ bp.messages[ne,from] for ne in from.neighbors ]
    # ignore message from destination
    μ_in[to] .= 1.0 
    # compute product of incoming messages and factor, and sum-out destination variable
    for x in CartesianIndices(axes(from.factor))
        μ[x[to]] += from.factor[x] * mapreduce(p -> p[2][x[p[1]]], *, enumerate(μ_in))    # mapreduce( pair -> pair[1] == to ? 1.0 : bp.messages[pair[2],from][x[pair[1]]], *, enumerate(from.neighbors))
    end    
    μ
end

"Compute belief propagation messages for each edge in factor graph."
function update!(bp::BeliefPropagation)
    # compute messages from factor to variable
    for f in bp.fg.factors, i in eachindex(f.neighbors)
        update!(bp,f,i)
    end
    # compute messages from factors to variables
    for v in bp.fg.variables, f in v.neighbors
        update!(bp,v,f)
    end
# for ((from,to),μ) in bp.messages
    #     println(typeof(from), "->", typeof(to), ": ", μ)
    # end    
    nothing
end

"Compute marginal distribution of given variable node from belief propagation messages."
function marginal(var::VariableNode, bp::BeliefPropagation)
    marginal = ones(length(var.variable))
    # multiply incoming messages
    for factor in var.neighbors
        marginal .*= bp.messages[factor,var]
    end
    # then normalize vector
    marginal ./= sum(marginal)
end
