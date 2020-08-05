# Sum-Product Belief Propagation for Marginal and LogPartition Computation
 
abstract type MessagePassingAlgorithm end
"Data structure for sum-product belief propagation algorithm."
mutable struct BeliefPropagation <: MessagePassingAlgorithm
    fg::FactorGraph
    iterations::Int
    λ::Float64 # dampening factor ∈ [0,1]
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    evidence::Dict{VariableNode,UInt}
    "Initialize belief propagation with no evidence."
    function BeliefPropagation(fg::FactorGraph) 
        # initialize messages as vectos of ones
        μ = Dict{Tuple{FGNode,FGNode},AbstractVector}()
        for v in values(fg.variables), f in v.neighbors
            # μ[v,f] = ones(length(v.variable)) # linear domain
            μ[v,f] = zeros(v.dimension) # log domain
        end
        for f in values(fg.factors), v in f.neighbors
            # μ[f,v] = ones(length(v.variable)) # linear domain
            μ[f,v] = zeros(v.dimension) # log domain
        end        
        new(fg, 0, 1.0, μ, Dict{VariableNode,UInt}())
    end
end

Base.getindex(bp::BeliefPropagation, from::FGNode, to::FGNode) = Base.getindex(bp.messages, (from,to))
Base.setindex!(bp::BeliefPropagation, μ, from::FGNode, to::FGNode) = Base.setindex!(bp.messages, μ, (from,to))

"Set evidence value on variable identified by id."
setevidence!(bp::BeliefPropagation, id::String, value) = bp.evidence[bp.fg.variables[id]] = value
"Removes evidence from variable identified by id."
rmevidence!(bp::BeliefPropagation, id::String) = delete!(bp.evidence,id)

"Removes any evidence and resets messages to their initial values (equal to zero for log-domain)."
function reset!(bp::BeliefPropagation) 
    empty!(bp.evidence)
    for μ in values(bp.messages)
        fill!(μ, 0.0)
    end
    nothing
end

"Update belief propagation message from variable to factor node. Returns residual."
function update!(bp::BeliefPropagation, from::VariableNode, to::FactorNode)
    # fill!(μ,1.0) # linear domain
    if haskey(bp.evidence,from) # variable is clamped, send indicator function
        e = bp.evidence[from]
        μ = bp[from,to]
        fill!(μ,-Inf) # log domain
        μ[e] = 0.0
        return 0.0
    end   
    μ = similar(bp[from,to])
    fill!(μ,0.0) # log domain
    # compute product of incoming messages
    for factor in from.neighbors
        if factor ≠ to
            # μ .*= bp.messages[factor,from] # linear domain
            μ .+= bp[factor,from] # incoming message in log domain
        end
    end    
    # # normalize message (make sum = 1)
    # μ .-= sum(μ)/length(μ)
    # compute residual (in log domain, should we compute it in linear domain?)
    # res = mapreduce(i ->  abs(bp.messages[from,to][i] - μ[i]), max, 1:length(μ)) # is this faster?
    # res = maximum(abs.(bp[from,to].-μ)) # is this slower?
    res = 0.0
    ϕ = bp[from,to]
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    # damped update
    # @. bp[from,to] = bp.λ*μ + (1.0-bp.λ)*bp[from,to]
    ϕ .= μ
    res
end

"Update belief propagation message from factor to its ith neighbor. Returns residual"
function update!(bp::BeliefPropagation, from::FactorNode, to::Integer)
    ϕ = copy(from.factor)
    dims = [ dim for dim in size(ϕ) ]
    # eliminate one neighbor at a time (do binary sum-product operations)
    for i = 1:(to-1) # eliminate from left up to to (not included)
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp[ne,from]             # incoming message
        m = maximum(ϕ) + maximum(μ)
        dim = popfirst!(dims)
        # sum-product: ψ(y) = ∑x ϕ(x,y) * μ(x)
        ψ = zeros(dims...)
        for x = 1:dim, y in CartesianIndices(axes(ψ))
            ψ[y] += exp(ϕ[x,y] + μ[x] - m)
        end
        ϕ = m .+ log.(ψ)
    end
    for i = length(from.neighbors):-1:(to+1) # eliminate from rightmost up to to (not included)
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp.messages[ne,from]    # incoming message
        m = maximum(ϕ) + maximum(μ)
        dim = pop!(dims)
        # sum-product: ψ(x) = ∑y ϕ(x,y) * μ(y)
        ψ = zeros(dims...)
        for x in CartesianIndices(axes(ψ)), y = 1:dim
            ψ[x] += exp(ϕ[x,y] + μ[y] - m)
        end
        ϕ = m .+ log.(ψ)
    end
    @assert length(ϕ) == from.neighbors[to].dimension "Got: $(length(ϕ)) Exp: $(from.neighbors[to].dimension)"
    # normalize message (make sum equal to one)
    ϕ .-= sum(ϕ[isfinite.(ϕ)])/length(ϕ)
    # compute residual
    # res = maximum(abs.(bp[from,from.neighbors[to]].-ϕ)) # is this slow?
    μ = bp.messages[from,from.neighbors[to]]
    res = 0.0
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    if bp.λ < 1 # damped update
        @. ϕ = bp.λ*ϕ + (1.0-bp.λ)*μ
    end
    #ϕ[isnan.(ϕ)] .= -Inf
    # @assert sum(isnan.(ϕ)) == 0
    μ .= ϕ
    res
    # # linear domain
    # μ = bp.messages[from,from.neighbors[to]]
    # fill!(μ,0.0)
    # # collect incoming messages other than from destination
    # μ_in = [ i == to ? ones(length(from.neighbors[to].variable)) : bp.messages[from.neighbors[i],from] for i=1:length(from.neighbors) ] # linear domain
    # # compute product of incoming messages and factor, and project on destination variable
    # for x in CartesianIndices(axes(from.factor))
    #     # μ[x[to]] += from.factor[x] * mapreduce(p -> p[2][x[p[1]]], *, enumerate(μ_in)) # linear domain
    #     # # μ[x[to]] += from.factor[x] * mapreduce( pair -> pair[1] == to ? 1.0 : bp.messages[pair[2],from][x[pair[1]]], *, enumerate(from.neighbors))
    # end    
    # μ
end

"Compute belief propagation messages for each edge in factor graph."
function update!(bp::BeliefPropagation)
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

"Compute marginal distribution of given variable node from belief propagation messages."
marginal(id::String, bp::BeliefPropagation) = marginal(bp.fg.variables[id], bp)
function marginal(var::VariableNode, bp::BeliefPropagation)
    # marginal = ones(length(var.variable)) # linear domain
    marginal = zeros(var.dimension) # log domain
    if haskey(bp.evidence, var)
        marginal[bp.evidence[var]] = 1.0
        return marginal
    end 
    # multiply incoming messages
    for factor in var.neighbors
        # marginal .*= bp.messages[factor,var] # linear domain
        marginal .+= bp[factor,var] # log domain
    end
    # then normalize vector
    # return marginal ./= sum(marginal) # linear domain
    # log domain
    # use log-sum-exp trick to avoid numerical errors
    marginal .-= maximum(marginal)
    marginal .= exp.(marginal)
    marginal ./= sum(marginal)
end
