@testset "BeliefPropagation" begin
    import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, marginal, decode
    @testset "A-B Tree" begin
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
        push!(bp.mapvars, x1)
        push!(bp.mapvars, x2)
        push!(bp.mapvars, x3)
        push!(bp.mapvars, x4)
        # Run message passing until convergence
        # while update!(bp) > 1e-10 end
        for i=1:10
            res = update!(bp)
            println("$i \t $res")
            if res < 1e-8 break end
        end            
        @info "converged in $(bp.iterations) iterations."
        # @test bp.iterations < 10
        for x in (x1,x2,x3,x4)
            println(x, " ", marginal(x,bp), " ", decode(x,bp), " ", x in bp.mapvars)
        end
        # check marginals    
    end
end