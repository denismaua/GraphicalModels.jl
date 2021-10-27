@testset "HybridBeliefPropagation" begin
    import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, setmapvar!, marginal, decode
    @testset "MaxProduct Inference in Bayes Tree" begin
        # Tree-shaped factor graph
        x4 = VariableNode(2)
        x3 = VariableNode(2)
        x2 = VariableNode(2)
        x1 = VariableNode(2)
        f1 = FactorNode(log.([0.3 0.6; 0.7 0.4]), VariableNode[x1, x3]) # P(X1|X3)
        f2 = FactorNode(log.([0.5 0.1; 0.5 0.9]), VariableNode[x2, x3]) # P(X2|X3)
        f3 = FactorNode(log.([0.2 0.7; 0.8 0.3]), VariableNode[x3, x4]) # P(X3|X4)
        f4 = FactorNode(log.([0.6, 0.4]), VariableNode[x4]) # P(X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph(
            Dict( "X1" => x1, "X2" => x2, "X3" => x3, "X4" => x4 ),
            Dict( "P(X1|X3)" => f1, "P(X2|X3)" => f2, "P(X3|X4)" => f3, "P(X4)" => f4 )
        )
        # Initialize hybrid belief propagation algorithm
        bp = HybridBeliefPropagation(fg)
        setmapvar!(bp, "X1")
        setmapvar!(bp, "X2")
        setmapvar!(bp, "X3")
        setmapvar!(bp, "X4")
        # push!(bp.mapvars, x1)
        # push!(bp.mapvars, x2)
        # push!(bp.mapvars, x3)
        # push!(bp.mapvars, x4)
        messages = [(x1,f1), (x2,f2), (f1,x3), (f2,x3), (x3,f3), (f3,x4), (f4,x4), (x4,f3), (f3,x3), (x3,f1), (x3,f2), (f1,x1), (f2,x2)]
        for (f,t) in messages
            update!(bp,f,t)
            # @show exp.(bp[f,t])
        end
        # Run message passing until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 3
        # check marginals    
        marginals = Dict(
            x1 => [0.6, 0.4],
            x2 => [0.27435610302351626, 0.7256438969764838],
            x3 => [0.27435610302351626, 0.7256438969764838],
            x4 => [0.7256438969764838, 0.27435610302351626]
            )    
        @testset "Checking marginal for $i" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(bp,x)
        end
    end
    @testset "MaxSumProduct Inference in Bayes Tree" begin
        # Tree-shaped factor graph
        x4 = VariableNode(2)
        x3 = VariableNode(2)
        x2 = VariableNode(2)
        x1 = VariableNode(2)
        f1 = FactorNode(log.([0.3 0.6; 0.7 0.4]), VariableNode[x1, x3]) # P(X1|X3)
        f2 = FactorNode(log.([0.5 0.1; 0.5 0.9]), VariableNode[x2, x3]) # P(X2|X3)
        f3 = FactorNode(log.([0.2 0.7; 0.8 0.3]), VariableNode[x3, x4]) # P(X3|X4)
        f4 = FactorNode(log.([0.6, 0.4]), VariableNode[x4]) # P(X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph(
            Dict( "X1" => x1, "X2" => x2, "X3" => x3, "X4" => x4 ),
            Dict( "P(X1|X3)" => f1, "P(X2|X3)" => f2, "P(X3|X4)" => f3, "P(X4)" => f4 )
        )
        # Initialize hybrid belief propagation algorithm
        bp = HybridBeliefPropagation(fg)    
        setmapvar!(bp,"X3")
        setmapvar!(bp,"X4")
        # push!(bp.mapvars, x3)
        # push!(bp.mapvars, x4)
        messages = [
            (x1,f1), # 1
            (x2,f2), # 2
            (f1,x3), # 3
            (f2,x3), # 4
            (x3,f3), # 5
            (f3,x4), # 6
            (f4,x4), # 7
            (x4,f3), # 8
            (f3,x3), # 9
            (x3,f1), # 10
            (x3,f2), # 11
            (f1,x1), # 12
            (f2,x2)] # 13
        for (f,t) in messages
            update!(bp,f,t)
            # @show exp.(bp[f,t])
        end
        # Run message passing until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 3
        # check marginals    
        marginals = Dict(
            x1 => [0.6, 0.4],
            x2 => [0.1, 0.9],
            x3 => [0.3684210526315789, 0.6315789473684211],
            x4 => [0.6315789473684211, 0.3684210526315789]
            )    
        @testset "Checking marginal for $i" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(bp,x)
        end
    end
end