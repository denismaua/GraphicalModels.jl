# Message Passing for Marginal and LogPartition Computation
 
abstract type MessagePassingAlgorithm end
"Data structure for sum-product belief propagation algorithm."
mutable struct BeliefPropagation <: MessagePassingAlgorithm
    fg::FactorGraph
    iterations::Int
    λ::Float64 # dampening factor ∈ [0,1]
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    "Initialize belief propagation with no evidence."
    function BeliefPropagation(fg::FactorGraph) 
        # initialize messages as vectos of ones
        μ = Dict{Tuple{FGNode,FGNode},AbstractVector}()
        for v in fg.variables, f in v.neighbors
            # μ[v,f] = ones(length(v.variable)) # linear domain
            μ[v,f] = zeros(length(v.variable)) # log domain
        end
        for f in fg.factors, v in f.neighbors
            # μ[f,v] = ones(length(v.variable)) # linear domain
            μ[f,v] = zeros(length(v.variable)) # log domain
        end        
        new(fg, 0, 1.0, μ)
    end
end

"Update belief propagation message from variable to factor node. Returns residual."
function update!(bp::BeliefPropagation, from::VariableNode, to::FactorNode)
    μ = similar(bp.messages[from,to])
    # fill!(μ,1.0) # linear domain
    fill!(μ,0.0) # log domain
    # compute product of incoming messages
    for factor in from.neighbors
        if factor ≠ to
            # μ .*= bp.messages[factor,from] # linear domain
            μ .+= bp.messages[factor,from] # log domain
        end
    end    
    # normalize message (make sum = 1)
    μ .-= sum(μ)/length(μ)
    # compute residual (in log domain, should we compute it in linear domain?)
    # res = mapreduce(i ->  abs(bp.messages[from,to][i] - μ[i]), max, 1:length(μ)) # is this faster?
    res = maximum(abs.(bp.messages[from,to].-μ)) # is this slower?
    # damped update
    @. bp.messages[from,to] = bp.λ*μ + (1.0-bp.λ)*bp.messages[from,to]
    res
end

"Update belief propagation message from factor to its ith neighbor. Returns residual"
function update!(bp::BeliefPropagation, from::FactorNode, to::Integer)
    ϕ = copy(from.factor)
    dims = [ dim for dim in size(ϕ) ]
    # eliminate one neighbor at a time (do binary sum-product operations)
    for i = 1:(to-1) # eliminate from left up to to (not included)
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp.messages[ne,from]    # incoming message
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
    @assert length(ϕ) == length(from.neighbors[to].variable) "Got: $(length(μ)) Exp: $(length(from.neighbors[to].variable))"
    # normalize message (make sum equal to one)
    ϕ .-= sum(ϕ)/length(ϕ)
    # compute residual
    res = maximum(abs.(bp.messages[from,from.neighbors[to]].-ϕ)) # is this slow?
    # TODO: damped update
    # @. bp.messages[from,from.neighbors[to]] = bp.λ*ϕ + (1.0-bp.λ)*bp.messages[from,from.neighbors[to]]
    bp.messages[from,from.neighbors[to]] .= ϕ
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
    # compute messages from factor to variable
    res = 0.0 # residual
    for f in bp.fg.factors, i in eachindex(f.neighbors)
        res = max(res,update!(bp,f,i))
    end
    # compute messages from factors to variables
    for v in bp.fg.variables, f in v.neighbors
        res = max(res,update!(bp,v,f))
    end
# for ((from,to),μ) in bp.messages
    #     println(typeof(from), "->", typeof(to), ": ", μ)
    # end    
    bp.iterations += 1
    res
end

"Compute marginal distribution of given variable node from belief propagation messages."
function marginal(var::VariableNode, bp::BeliefPropagation)
    # marginal = ones(length(var.variable)) # linear domain
    marginal = zeros(length(var.variable)) # log domain
    # multiply incoming messages
    for factor in var.neighbors
        # marginal .*= bp.messages[factor,var] # linear domain
        marginal .+= bp.messages[factor,var] # log domain
    end
    # then normalize vector
    # return marginal ./= sum(marginal) # linear domain
    # log domain
    # use log-sum-exp trick to avoid numerical errors
    marginal .-= maximum(marginal)
    marginal .= exp.(marginal)
    marginal ./= sum(marginal)
end
