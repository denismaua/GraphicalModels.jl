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
    VariableNode(d::Integer) = new(d, FactorNode[])  
    VariableNode(v::Variable) = new(v.dimension, FactorNode[])  
end
Base.show(io::IO,v::VariableNode) = print(io,"Variable(dim=$(v.dimension), degree=$(length(v.neighbors)))") 
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


"""
    FactorGraph(filename::AbstractString)::FactorGraph
    FactorGraph(io::IO=stdin)::FactorGraph

Reads a model from file in UAI Competition Format and returns the correspoding factor graph.
See https://www.cs.huji.ac.il/project/UAI10/fileFormat.php for details about the file format.
"""
function FactorGraph(filename::String)
    spn = open(filename) do file
        spn = FactorGraph(file)
    end
    spn
end
function FactorGraph(io::IO=stdin)
    # create dictionaries of variables and factors
    vars = Dict{String,VariableNode}()
    factors = Dict{String,FactorNode}()
    # read and create nodes
    state = "header"
    numvars = 0
    numfactors = 0
    nfread = 0 # no. of factors read
    szfactor = 0 # no. of values to read from current factor
    factor = Float64[] # temporary factor values 
    for line in eachline(io)
        # remove line break
        line = strip(line)
        # remove comments
        i = findfirst(isequal('#'), line)
        if !isnothing(i)
            line = line[1:i-1]
        end
        if length(line) > 0         
            nothing
            # State machine
            if state == "header"
                # parse header: MARKOV or BAYES (warn otherwise)
                if !startswith(line, "MARKOV") && !startswith(line,"BAYES")
                    @warn "Unrecognized header: $line"
                end
                state = "numvars"
            elseif state == "numvars"
                numvars = parse(Int,line)
                # @info "$numvars variables."
                sizehint!(vars,numvars)
                state = "numstates"
            elseif state == "numstates"
                fields = split(line)
                for (i,value) in enumerate(fields)
                    vars[string(i-1)] = VariableNode(parse(Int,value))
                end
                state = "numfactors"
            elseif state == "numfactors"
                numfactors = parse(Int,line)
                sizehint!(factors,numfactors)
                # @info "$numfactors factors."
                state = "scopes"
            elseif state == "scopes"
                fields = split(line)
                size = parse(Int,fields[1]) # no. of variables in scope
                neighbors = map( f -> vars[f], fields[2:end] )
                fdims = map( v -> v.dimension, neighbors )
                factors[string(nfread)] = FactorNode(Array{Float64}(undef,fdims...), neighbors)
                nfread += 1
                if nfread == numfactors
                    nfread = 0
                    state = "factors"
                end
            elseif state == "factors"
                fields = split(line)
                i = 1
                if szfactor == 0
                    szfactor = parse(Int,fields[1]) # no. of values in scope            
                    @assert szfactor == length(factors[string(nfread)].factor)
                    sizehint!(factor,szfactor)
                    i = 2
                end                
                append!(factor, parse.(Float64,fields[i:end]))
                if szfactor == length(factor) # end of factor values
                    ϕ = factors[string(nfread)]
                    rfdims = Tuple(v.dimension for v in Iterators.reverse(ϕ.neighbors))
                    ϕ.factor .= log.(permutedims(reshape(factor,rfdims), ndims(ϕ.factor):-1:1 ))
                    # @info nfread ϕ.factor
                    factor = empty(factor)
                    szfactor = 0                   
                    nfread += 1
                    if nfread == numfactors
                        state = "end"
                    end
                end
            elseif state == "end"
                @warn "Found unknown extra content in file."
            end
            # header -> no. of variables -> list of vars cardinalities -> no. of factors -> scopes -> factors
        end
    end   
    # create and return model 
    FactorGraph(vars, factors)
end