# Max-Sum-Product Belief Propagation for Marginal MAP Inference.
# Based on the work of "Liu and Ihler, Variational Algorithms for Marginal MAP, J. Mach. Learning Res. 2013."
 
"Data structure for max-sum-product belief propagation algorithm."
mutable struct HybridBeliefPropagation <: MessagePassingAlgorithm
    fg::FactorGraph
    iterations::Int
    λ::Float64 # dampening factor ∈ [0,1]
    normalize::Bool # normalize messages?
    messages::Dict{Tuple{FGNode,FGNode},AbstractVector}
    evidence::Dict{VariableNode,UInt}
    mapvars::Set{VariableNode}
    "Initialize belief propagation with noninformative messages; set rndinit = true to use random initialization."
    function HybridBeliefPropagation(fg::FactorGraph; rndinit=false) 
        # initialize messages as vectos of ones
        μ = Dict{Tuple{FGNode,FGNode},AbstractVector}()
        for v in values(fg.variables), f in v.neighbors
            # μ[v,f] = ones(length(v.variable)) # linear domain
            if rndinit
                μ[v,f] = 0.1*randn(v.dimension)
            else
                μ[v,f] = zeros(v.dimension) # log domain
            end
        end
        for f in values(fg.factors), v in f.neighbors
            # μ[f,v] = ones(length(v.variable)) # linear domain
            if rndinit
                μ[f,v] = 0.1*randn(v.dimension)
            else
                μ[f,v] = zeros(v.dimension) # log domain
            end
        end        
        new(fg, 0, 1.0, false, μ, Dict{VariableNode,UInt}(), Set{VariableNode}())
    end
end

"""
    setmapvar!(mp::MessagePassingAlgorithm, id::String)

Assign variable with given `id` to be maximized.
"""
setmapvar!(bp::HybridBeliefPropagation, id::String) = push!(bp.mapvars, bp.fg.variables[id])
"Removes evidence from variable identified by `id`."
unsetmapvar!(bp::HybridBeliefPropagation, id::String) = delete!(mp.evidence, bp.fg.variables[id])


"Update hybrid belief propagation message `from` variable `to` factor node. Returns residual."
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
update!(bp::HybridBeliefPropagation, from::FactorNode, to::VariableNode) = update!(bp,from,findfirst(isequal(to),from.neighbors))
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
                # ψ[y] = m + log(sum(x -> exp(ϕ[x,y] + μ[x] - m), 1:dim))
                ψ[y] = log(sum(x -> exp(ϕ[x,y] + μ[x] - m), 1:dim))
            end
        else # ne is MAP node
            if vto ∈ bp.mapvars # map to map 
                # do max-product update: ψ(y) = max_x ϕ(x,y) * μ(x)
                for y in CartesianIndices(axes(ψ))
                    ψ[y] = maximum(x -> (ϕ[x,y] + μ[x]), 1:dim)
                end
            else
                # do argmax-product update: ψ(y) =  ϕ(X,y) * μ(X) where X = argmax μ(x)*bp[from,ne](x)
                # m = maximum(ϕ) + maximum(μ)
                μ_ne = μ .+ bp[from,ne]
                # x = argmax(μ_ne)
                mx = maximum(μ_ne)
                X = findall(m -> isapprox(m,mx), μ_ne) # argmax_x μ_ne(x)
                for y in CartesianIndices(axes(ψ))
                    # ψ[y] = m + log(sum(x -> exp(ϕ[x,y] + μ[x] - m), X))
                    # ψ[y] = log(sum(x -> exp(ϕ[x,y] + μ[x] - m), X))
                    ψ[y] = log(sum(x -> exp(ϕ[x,y] + μ[x]), X))
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
            for x in CartesianIndices(axes(ψ))
                # ψ[x] = m + log(sum(y -> exp(ϕ[x,y] + μ[y] - m), 1:dim))
                ψ[x] = log(sum(y -> exp(ϕ[x,y] + μ[y] - m), 1:dim))
            end
        else # ne is MAP node
            if vto ∈ bp.mapvars # MAP to MAP
                # do max-product update: ψ(x) = max_y ϕ(x,y) * μ(y)
                for x in CartesianIndices(axes(ψ))
                    ψ[x] = maximum(y -> begin ϕ[x,y] + μ[y] end, 1:dim)
                end
            else
                # do argmax-product update: ψ(x) =  ϕ(x,Y) * μ(Y) where Y = argmax μ(y)*bp[from,ne](y)
                # m = maximum(ϕ) + maximum(μ)
                μ_ne = μ .+ bp[from,ne]
                # y = argmax(μ_ne)
                my = maximum(μ_ne)
                Y = findall(y -> isapprox(y,my), μ_ne) # argmax_y μ_ne(y)
                for x in CartesianIndices(axes(ψ))
                    # ψ[x] = m + log(sum(y -> exp(ϕ[x,y] + μ[y] - m), Y))
                    ψ[x] = log(sum(y -> exp(ϕ[x,y] + μ[y]), Y))
                end
            end
        end
        ϕ = ψ
    end
    @assert length(ϕ) == vto.dimension "Got: $(length(ϕ)) Exp: $(vto.dimension)"
    if bp.normalize # normalize message (make sum equal to one in linear domain)
        # ϕ .-= sum(ϕ[isfinite.(ϕ)])/length(ϕ)
        ϕ .-= maximum(ϕ)
        ϕ .-= log(mapreduce(exp,+,ϕ))
    end
    # compute residual
    μ = bp.messages[from,vto]
    @assert length(ϕ) == length(μ)
    res = 0.0
    for i=1:length(μ)
        if isfinite(μ[i]) && isfinite(ϕ[i])
            res = max(res, abs(μ[i]-ϕ[i]))
        else
            res = max(res, max(μ[i], ϕ[i]))
        end
    end
    if bp.λ < 1 # damped update
        for i=1:length(ϕ) # prevents resulting with -Inf messages
            if isfinite(μ[i]) && isfinite(ϕ[i])
                ϕ[i] = bp.λ * ϕ[i] + (1.0-bp.λ) * μ[i]
            end            
        end
    end
    @assert count(isfinite.(ϕ)) > 0 "Not finite: $ϕ"
    μ .= ϕ
    res
end

"""
    marginal(bp::HybridBeliefPropagation, id::String)
    marginal(bp::HybridBeliefPropagation, var::VariableNode)

Compute marginal distribution of given variable node from hybrid belief propagation messages.
"""
marginal(bp::HybridBeliefPropagation, id::String) = marginal(bp, bp.fg.variables[id])
function marginal(bp::HybridBeliefPropagation, var::VariableNode)
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

"""
    decode(bp::HybridBeliefPropagation, id::String)
    decode(bp::HybridBeliefPropagation, var::VariableNode)

Compute maximizing value for marginal belief of given variable.
"""
decode(bp::HybridBeliefPropagation, id::String) = decode(bp, bp.fg.variables[id])
function decode(bp::HybridBeliefPropagation, var::VariableNode)
    if haskey(bp.evidence, var)
        return bp.evidence[var]
    end     
    marginal = zeros(var.dimension) 
    for factor in var.neighbors
        marginal += bp[factor,var] 
    end
    argmax(marginal)
end