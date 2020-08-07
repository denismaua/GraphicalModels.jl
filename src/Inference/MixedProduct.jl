# Max-Sum-Product Belief Propagation for Marginal MAP Inference.
# Based on the work of "Liu and Ihler, Variational Algorithms for Marginal MAP, J. Mach. Learning Res. 2013."
 
"Data structure for max-sum-product belief propagation algorithm."
mutable struct HybridBeliefPropagation <: MessagePassingAlgorithm
    fg::FactorGraph
    iterations::Int
    λ::Float64 # dampening factor ∈ [0,1]
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    evidence::Dict{VariableNode,UInt}
    mapvars::Set{VariableNode}
    "Initialize belief propagation with no evidence."
    function HybridBeliefPropagation(fg::FactorGraph) 
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
        new(fg, 0, 1.0, μ, Dict{VariableNode,UInt}(), Set{VariableNode}())
    end
end

"Update hybrid belief propagation message from variable to factor node. Returns residual."
function update!(bp::HybridBeliefPropagation, from::VariableNode, to::FactorNode)
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
            μ .+= bp[factor,from] # incoming message in log domain
        end
    end    
    res = 0.0
    ϕ = bp[from,to]
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    ϕ .= μ
    res
end

"Update hybrid belief propagation message from factor to its ith neighbor. Returns residual"
function update!(bp::HybridBeliefPropagation, from::FactorNode, to::Integer)
    ϕ = copy(from.factor)
    vto = from.neighbors[to] # destination variable
    dims = [ dim for dim in size(ϕ) ]
    # eliminate one neighbor at a time (do binary max-sum-product operations)
    for i = 1:(to-1) # eliminate from left up to to (not included)
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp[ne,from]             # incoming message
        dim = popfirst!(dims)
        ψ = Array{Float64}(undef,dims...)
        if ne ∉ bp.mapvars # ne is sum node 
            # do sum-product update: ψ(y) = ∑_x ϕ(x,y) * μ(x)
            m = maximum(ϕ) + maximum(μ)
            for y in CartesianIndices(axes(ψ))
                ψ[y] = m + log(sum(x -> exp(ϕ[x,y] + μ[x] - m), 1:dim))
            end
        else # ne is MAP node
            if vto ∈ bp.mapvars # map to map 
                # do max-product update: ψ(y) = max_x ϕ(x,y) * μ(x)
                for y in CartesianIndices(axes(ψ))
                    ψ[y] = maximum(x -> (ϕ[x,y] + μ[x]), 1:dim)
                end
            else
                # do argmax-product update: ψ(y) =  ϕ(X,y) * μ(X) where X = argmax μ(x)*bp[from,ne](x)
                m = maximum(ϕ) + maximum(μ)
                μ_ne = μ .* bp[from,ne]
                # x = argmax(μ_ne)
                mx = maximum(μ_ne)
                X = findall(x -> isapprox(x,mx), μ_ne) # argmax_x μ_ne(x)
                for y in CartesianIndices(axes(ψ))
                    ψ[y] = m + log(sum(x -> exp(ϕ[x,y] + μ[x] - m), X))
                end
            end
        end
        ϕ = ψ
    end
    for i = length(from.neighbors):-1:(to+1) # eliminate from rightmost up to to (not included)
        ne = from.neighbors[i]      # i-th neighbor node
        μ = bp.messages[ne,from]    # incoming message
        dim = pop!(dims)
        ψ = Array{Float64}(undef,dims...)
        if ne ∉ bp.mapvars # ne is sum node 
            # do sum-product update: ψ(x) = ∑_y ϕ(x,y) * μ(y)
            m = maximum(ϕ) + maximum(μ)
            for x in CartesianIndices(axes(ψ)), y = 1:dim
                ψ[x] = m + log(sum(y -> exp(ϕ[x,y] + μ[y] - m),1:dim))
            end
        else # ne is MAP node
            if vto ∈ bp.mapvars # MAP to MAP
                # do max-product update: ψ(x) = max_y ϕ(x,y) * μ(y)
                for x in CartesianIndices(axes(ψ))
                    ψ[x] = maximum(y -> begin ϕ[x,y] + μ[y] end, 1:dim)
                end
            else
                # do argmax-product update: ψ(x) =  ϕ(x,Y) * μ(Y) where Y = argmax μ(y)*bp[from,ne](y)
                m = maximum(ϕ) + maximum(μ)
                μ_ne = μ .* bp[from,ne]
                # y = argmax(μ_ne)
                my = maximum(μ_ne)
                Y = findall(y -> isapprox(y,my), μ_ne) # argmax_y μ_ne(y)
                for x in CartesianIndices(axes(ψ))
                    ψ[x] = m + log(sum(y -> exp(ϕ[x,y] + μ[y] - m), Y))
                end
            end
        end
        ϕ = ψ
    end
    @assert length(ϕ) == vto.dimension "Got: $(length(ϕ)) Exp: $(vto.dimension)"
    # normalize message (make sum equal to one)
    # ϕ .-= sum(ϕ[isfinite.(ϕ)])/length(ϕ)
    # compute residual
    μ = bp.messages[from,vto]
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
    μ .= ϕ
    res
end

"Compute marginal distribution of given variable node from hybrid belief propagation messages."
marginal(id::String, bp::HybridBeliefPropagation) = marginal(bp.fg.variables[id], bp)
function marginal(var::VariableNode, bp::HybridBeliefPropagation)
    marginal = zeros(var.dimension) # log domain
    if haskey(bp.evidence, var)
        marginal[bp.evidence[var]] = 1.0
        return marginal
    end 
    # multiply incoming messages
    for factor in var.neighbors
        marginal += bp[factor,var] # log domain
    end
    # then normalize vector
    marginal .-= maximum(marginal)
    marginal .= exp.(marginal)
    marginal ./= sum(marginal)
end

"Compute maximizing value for local belief."
decode(id::String, bp::HybridBeliefPropagation) = decode(bp.fg.variables[id], bp)
function decode(var::VariableNode, bp::HybridBeliefPropagation)
    if haskey(bp.evidence, var)
        return bp.evidence[var]
    end     
    marginal = zeros(var.dimension) 
    for factor in var.neighbors
        marginal += bp[factor,var] 
    end
    argmax(marginal)
end