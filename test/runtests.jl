using GraphicalModels
import GraphicalModels: VariableNode, FactorNode, FactorGraph
import GraphicalModels: BeliefPropagation, update!, marginal
using Test

@testset "BeliefPropagation" begin
    @testset "Exact Inference in Bayesian Trees" begin
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
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 10
        # for x in (x1,x2,x3,x4)
        #     println(x.variable, " ", marginal(x,bp))
        # end
        # check marginals
        marginals = Dict(
            x1 => [0.48, 0.52],
            x2 => [0.26, 0.74],
            x3 => [0.4, 0.6],
            x4 => [0.6, 0.4]
            )
        @testset "Checking marginal for $i" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(x,bp)
        end
    end # end of Bayesian Tree testset
    @testset "Exact Inference in Bayes Tree with evidence" begin
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
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)  
        # Set some evidence
        bp.evidence[x1] = 1
        bp.evidence[x2] = 2
        # Run belief propagation for until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 10        
        marginals = Dict(
            x1 => [1.0, 0.0],
            x2 => [0.0, 1.0],
            x3 => [0.15625, 0.84375],
            x4 => [0.721875, 0.278125]
        )        
        @testset "Checking marginal for $i" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(x,bp)
            # println( "P($i) = $(marginal(x,bp))" )
        end
    end
    @testset "Exact Inference in Markov Trees" begin
        x1 = VariableNode(2)
        x2 = VariableNode(2)
        x3 = VariableNode(2)
        x4 = VariableNode(2)
        f12 = FactorNode(log.([10.0 0.1; 0.1 10.0]), VariableNode[x1, x2]) # phi(X1,X2)
        f1 = FactorNode(log.([5.0, 0.2]), VariableNode[x1]) # phi(X1)
        f3 = FactorNode(log.([1.0, 1.0]), VariableNode[x3]) # phi(X3)
        f24 = FactorNode(log.([5.0 0.2; 0.2 5.0]), VariableNode[x2, x4]) # phi(X2,X4)
        f34 = FactorNode(log.([0.5 20.0; 1.0 2.5]), VariableNode[x3, x4]) # phi(X3,X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([x1,x2,x3,x4],[f12,f1,f3,f24,f34])
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run BP until converegence
        while (update!(bp) > 1e-10) end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 10
        # check marginals
        marginals = Dict(
            x1 => [0.7440152339499456, 0.25598476605005444],
            x2 => [0.6803590859630033, 0.3196409140369967],
            x3 => [0.6521808124773303, 0.34781918752266955],
            x4 => [0.4260745375408052, 0.5739254624591947]
        )
        @testset "Checking marginal of $(i)" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(x,bp)
        end
    end # end of Markov tree testset
    @testset "Exact Inference in Tree-Decomposable Loopy Graph" begin
        x1 = VariableNode(2)
        x2 = VariableNode(2)
        x3 = VariableNode(2)
        x4 = VariableNode(2)
        f12 = FactorNode(log.([10 0.1; 0.1 10]), VariableNode[x1, x2]) # phi(X1,X2)
        f13 = FactorNode(log.([5 5; 0.2 0.2]), VariableNode[x1, x3]) # phi(X1,X3) = phi(X1)*phi(X3) of previous test
        f24 = FactorNode(log.([5 0.2; 0.2 5]), VariableNode[x2, x4]) # phi(X2,X4)
        f34 = FactorNode(log.([0.5 20; 1 2.5]), VariableNode[x3, x4]) # phi(X3,X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([x1,x2,x3,x4],[f12,f13,f24,f34])
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for succificient number of iterations
        while (update!(bp) > 1e-10 && bp.iterations < 10) nothing end
        @info "converged in $(bp.iterations) iterations."
        # check marginals
        marginals = Dict(
            x1 => [0.7440152339499456, 0.25598476605005444],
            x2 => [0.6803590859630033, 0.3196409140369967],
            x3 => [0.6521808124773303, 0.34781918752266955],
            x4 => [0.4260745375408052, 0.5739254624591947]
        )    
        @testset "Checking marginal for $(i)" for (i,x) in fg.variables
            @test marginals[x] ≈ marginal(x,bp)
        end    
    end # end of testset Easy Loopy
    @testset "Approximate Inference in Tree-Decomposable Loopy Graph" begin
        x1 = VariableNode(2)
        x2 = VariableNode(2)
        x3 = VariableNode(2)
        x4 = VariableNode(2)
        f12 = FactorNode(log.([10 0.1; 0.1 10]), VariableNode[x1, x2]) # phi(X1,X2)
        f13 = FactorNode(log.([5 0.2; 0.2 5]), VariableNode[x1, x3]) # phi(X1,X3) = phi(X1)*phi(X3) of previous test
        f24 = FactorNode(log.([5 0.2; 0.2 5]), VariableNode[x2, x4]) # phi(X2,X4)
        f34 = FactorNode(log.([0.5 20; 1 2.5]), VariableNode[x3, x4]) # phi(X3,X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([x1,x2,x3,x4],[f12,f13,f24,f34])
        # Ground truth
        marginals = Dict(
            x1 => [0.315508859965501, 0.684491140034499],
            x2 => [0.2767759134389211, 0.7232240865610788],
            x3 => [0.4699342689875072, 0.5300657310124928],
            x4 => [0.1207170299513878, 0.8792829700486121]
        )    
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for succificient number of iterations
        res = 0.0
        for i=1:100
            res = update!(bp)
            # compute mean absolute error
            mae = 0.0
            for x in values(fg.variables)
                mae += mapreduce(abs,+,marginal(x,bp) .- marginals[x])/2                
            end
            mae = mae/4
            computed = marginal(x1,bp)[1]
            expected = marginals[x1][1]
            @info "iteration: $(bp.iterations) \t residual: $(round(res;digits=6)) \t MAE: $(round(mae;digits=6)) \t P(X1=1): $(round(computed;digits=3)) [Error: $(computed-expected)]" maxlog=20
            if res < 1e-8 break end
            bp.λ *= 0.99 # exponential decay (damping)
        end
        # MAE should be small
        mae = 0.0
        for x in values(fg.variables)
            mae += mapreduce(abs,+,marginal(x,bp) .- marginals[x])/2
        end
        mae = mae/4
        computed = marginal(x1,bp)[1]
        expected = marginals[x1][1]
        @info "iteration: $(bp.iterations) \t residual: $(round(res;digits=6)) \t MAE: $(round(mae;digits=6)) \t P(X1=1): $(round(computed;digits=3)) [Error: $(computed-expected)]" 
        @test mae < 0.15
        # check marginals
        # @testset "Checking marginal for $(X.variable)" for (X,m) in marginals
        #     @test m ≈ marginal(X,bp)
        # end    
        # log partition
        # Z = 1224.384
    end # end of testset Hard Loopy    
end # end of testset BeliefPropagation
