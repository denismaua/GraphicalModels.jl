# Implemens Factor Graphics data type
export 
    VariableNode,
    FactorNode,
    FactorGraph

"A node of a Factor Graph."
abstract type FGNode end
"""
Represents a variable node.

# Arguments
- `variable`: integer identifying variable
- `neighbors`: adjacent factor nodes
"""
struct VariableNode <: FGNode
    variable::Int
    neighbors::Vector{FGNode}  
    VariableNode(v) = new(v,FactorNode[])  
end
"""
Representes a factor node.

# Arguments
- `factor`: a multidimensional array representing a function of the neighbors. Each dimension corresponds to the order of the variable in the vector neighbors
- `neighbors`: adjacent variable nodes
"""
struct FactorNode <: FGNode
    factor::AbstractArray
    neighbors::Vector{VariableNode}
    function FactorNode(f,ne)
        @assert ndims(f) == length(ne)
        new(f,ne)
    end
end
"""
A Factor Graph is a bipartite graph where nodes are either variables or factors.
# Arguments
- `variables`: vector of variable nodes.
- `factors`: vector of factor nodes.
"""
struct FactorGraph
    variables::Vector{VariableNode}
    factors::Vector{FactorNode}
    function FactorGraph(vars,factors)
        for factor in factors
            for v in factor.neighbors
                push!(v.neighbors,factor)
            end
        end
        new(vars,factors)
    end
end
