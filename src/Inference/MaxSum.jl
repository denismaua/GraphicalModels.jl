# Max-Sum Belief Propagation for Max-Marginal Computation
 
"Data structure for the max-sum belief propagation algorithm."
mutable struct MaxSum <: MessagePassingAlgorithm
    fg::FactorGraph
    iterations::Int
    λ::Float64 # damping factor ∈ [0,1]; 0 means no damping
    normalize::Bool # normalize messages?
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    evidence::Dict{VariableNode,UInt}
    "Initialize belief propagation with no evidence."
    function MaxSum(fg::FactorGraph) 
        # initialize messages as vectos of zeros
        μ = Dict{Tuple{FGNode,FGNode},AbstractVector}()
        for v in values(fg.variables), f in v.neighbors
            μ[v,f] = zeros(v.dimension) # log domain
        end
        for f in values(fg.factors), v in f.neighbors
            μ[f,v] = zeros(v.dimension) # log domain
        end        
        new(fg, 0, 0.0, true, μ, Dict{VariableNode,UInt}())
    end
end

"Update max-sum belief propagation message from variable to factor node and return residual."
function update!(bp::MaxSum, from::VariableNode, to::FactorNode)
    if haskey(bp.evidence,from) # variable is clamped, send indicator function
        e = bp.evidence[from]
        μ = bp[from,to]
        fill!(μ,-Inf) # log domain
        μ[e] = 0.0
        return 0.0
    end
    μ = similar(bp[from,to]) # message vector
    fill!(μ,0.0) # log domain
    # compute product of incoming messages
    for factor in from.neighbors
        if factor ≠ to
            μ .+= bp[factor,from] # incoming message in log domain
        end
    end    
    # normalize message (make sum = 0)
    if bp.normalize
        μ .-= sum(μ)/length(μ)
        #μ .-= sum(μ[isfinite.(μ)])/length(μ)
    end
    ϕ = bp[from,to] # points to previous message
    # Apply damping
    if bp.λ > 0
        @. μ = bp.λ*ϕ + (1-bp.λ)*μ
    end
    # Computes residual as supremum norm
    res = 0.0
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    # store updated message
    ϕ .= μ
    # return residual
    res
end

"Update max-sum belief propagation message from factor to variable and return residual."
update!(bp::MaxSum, from::FactorNode, to::VariableNode) = update!(bp,from,findfirst(isequal(to),from.neighbors))
"Update max-sum belief propagation message from factor to its i-th neighbor and return residual."
function update!(bp::MaxSum, from::FactorNode, to::Integer)
    ϕ = copy(from.factor)
    dims = [ dim for dim in size(ϕ) ]
    # Eliminate one neighbor variable at a time (do binary sum-product operations)
    # First eliminate variables that appear before (to left) of variable `to` in the factor's scope
    for i = 1:(to-1) 
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp[ne,from]             # incoming message
        dim = popfirst!(dims)
        # max-sum: ψ(y) = max_x ϕ(x,y) + μ(x)
        ψ = Array{Float64}(undef,dims...)
        for y in CartesianIndices(axes(ψ)) #, x = 1:dim
            ψ[y] = maximum(x -> ϕ[x,y] + μ[x], 1:dim)
        end
        ϕ = ψ
    end
    # Then eliminate variable that appear to the right of variable `to`
    for i = length(from.neighbors):-1:(to+1) 
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp.messages[ne,from]    # incoming message
        dim = pop!(dims)
        # max-sum: ψ(x) = max_y ϕ(x,y) + μ(y)
        ψ = Array{Float64}(undef,dims...)
        for x in CartesianIndices(axes(ψ)) #, y = 1:dim
            ψ[x] = maximum(y -> ϕ[x,y] + μ[y], 1:dim)
        end
        ϕ = ψ 
    end
    @assert length(ϕ) == from.neighbors[to].dimension "Got: $(length(ϕ)) Exp: $(from.neighbors[to].dimension)"
    if bp.normalize # normalize message (make sum equal to 0)
        ϕ .-= sum(ϕ)/length(ϕ)
        # ϕ .-= sum(ϕ[isfinite.(ϕ)])/length(ϕ)
    end
    # compute residual as max norm
    # res = maximum(abs.(bp[from,from.neighbors[to]].-ϕ)) # is this slow?
    μ = bp.messages[from,from.neighbors[to]] # previous message
    res = 0.0
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    # if bp.λ < 1 # damped update
    #     @. ϕ = bp.λ*ϕ + (1.0-bp.λ)*μ
    # end
    #ϕ[isnan.(ϕ)] .= -Inf
    # @assert sum(isnan.(ϕ)) == 0
    # Update message
    μ .= ϕ
    # return residual
    res
end

"""
    marginal(bp::MaxSum, id::String)
    marginal(bp::MaxSum, var::VariableNode)

Compute belief distribution of given variable node from max-sum belief propagation messages.
"""
marginal(bp::MaxSum, id::String) = marginal(bp, bp.fg.variables[id])
function marginal(bp::MaxSum, var::VariableNode)
    marginal = zeros(var.dimension) # log domain
    if haskey(bp.evidence, var)
        marginal[bp.evidence[var]] = 1.0
        return marginal
    end 
    # add up incoming messages
    for factor in var.neighbors
        marginal .+= bp[factor,var] # log domain
    end
    # then normalize vector
    marginal .-= maximum(marginal)
    marginal .= exp.(marginal)
    marginal ./= sum(marginal)    
    marginal
end

"""
    decode(bp::MaxSum, id::String)
    decode(bp::MaxSum, var::VariableNode)

Compute maximizing value of belief of given variable.
"""
decode(bp::MaxSum, id::String) = decode(bp, bp.fg.variables[id])
function decode(bp::MaxSum, var::VariableNode)
    if haskey(bp.evidence, var)
        return bp.evidence[var]
    end     
    marginal = zeros(var.dimension) 
    for factor in var.neighbors
        marginal += bp[factor,var] 
    end
    argmax(marginal)
end