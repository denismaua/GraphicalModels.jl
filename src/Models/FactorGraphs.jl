# Implemens Factor Graphics data type

"A node of a Factor Graph."
abstract type FGNode end
"""
Represents a variable node.

# Arguments
- `variable`: integer identifying variable
- `neighbors`: adjacent factor nodes
"""
struct VariableNode <: FGNode
    # variable::Variable
    dimension::UInt # no. of values of variable
    neighbors::Vector{FGNode}
    evidence::UInt  # 0 for no evidence or value ∈ [1,variable.dimension]
    VariableNode(d::Integer) = new(d, FactorNode[], 0)  
    VariableNode(v::Variable) = new(v.dimension, FactorNode[], 0)  
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
        # consistency checks
        @assert ndims(f) == length(ne)
        for (i,d) in enumerate(size(f))
            @assert d == ne[i].dimension
        end
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
    # variables::Vector{VariableNode}
    # factors::Vector{FactorNode}
    variables::Dict{String,VariableNode}
    factors::Dict{String,FactorNode}
    function FactorGraph(vars::Dict{String,VariableNode},factors::Dict{String,FactorNode})
        # Adds neighbors to variables
        for factor in values(factors)
            for v in factor.neighbors
                push!(v.neighbors,factor)
            end
        end
        new(vars,factors)
    end
    function FactorGraph(vars::Vector{VariableNode},factors::Vector{FactorNode})
        # Adds neighbors to variables
        for factor in factors
            for v in factor.neighbors
                push!(v.neighbors,factor)
            end
        end
        new(Dict( string(i) => v for (i,v) in enumerate(vars)), Dict( string(i) => f for (i,f) in enumerate(factors) ))
    end    
end

# "Sets evidence at variable node."
# function setfield!(fg::FactorGraph, X::VariableNode, x::Integer)
#     @assert x > 0 && x <= length(X.variable)
#     X.evidence = x
# end
