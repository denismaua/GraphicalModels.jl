# Implemens Factor Graphics data type
export 
    Variable,
    VariableNode,
    FactorNode,
    FactorGraph

"A Discrete Variable."
struct Variable
    id::Int
    dimension::UInt # no. of values
end
Base.show(io::IO,v::Variable) = print(io,"Variable(id=$(v.id), dim=$(v.dimension))") 
"Returns the number of values of the variable."
Base.length(v::Variable) = v.dimension

"A node of a Factor Graph."
abstract type FGNode end
"""
Represents a variable node.

# Arguments
- `variable`: integer identifying variable
- `neighbors`: adjacent factor nodes
"""
struct VariableNode <: FGNode
    variable::Variable
    neighbors::Vector{FGNode}
    evidence::UInt  # 0 for no evidence or value ∈ [1,variable.dimension]
    VariableNode(v) = new(v,FactorNode[], 0)  
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
            @assert d == ne[i].variable.dimension
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
    variables::Vector{VariableNode}
    factors::Vector{FactorNode}
    function FactorGraph(vars,factors)
        # Adds neighbors to variables
        for factor in factors
            for v in factor.neighbors
                push!(v.neighbors,factor)
            end
        end
        new(vars,factors)
    end
end
