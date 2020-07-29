using GraphicalModels
import GraphicalModels: BeliefPropagation, update!, marginal
using Test

@testset "BeliefPropagation" begin
    @testset "Exact Inference in Bayesian Trees" begin
        # Tree-shaped factor graph
        X4 = VariableNode(Variable(4,2))
        X3 = VariableNode(Variable(3,2))
        X2 = VariableNode(Variable(2,2))
        X1 = VariableNode(Variable(1,2))
        f1 = FactorNode([0.3 0.6; 0.7 0.4], VariableNode[X1, X3]) # P(X1|X3)
        f2 = FactorNode([0.5 0.1; 0.5 0.9], VariableNode[X2, X3]) # P(X2|X3)
        f3 = FactorNode([0.2 0.7; 0.8 0.3], VariableNode[X3, X4]) # P(X3|X4)
        f4 = FactorNode([0.6, 0.4], [X4]) # P(X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([X1,X2,X3,X4],[f1,f2,f3,f4])
        # for v in fg.variables
        #     println("$(v.variable)")
        #     for f in v.neighbors
        #         println("\t", f.factor)
        #     end
        # end
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for succificient number of iterations
        for i=1:4
            # println(i)
            # for X in (X1,X2,X3,X4)
            #     println(X.variable, " ", marginal(X,bp))
            # end
            update!(bp)
        end
        # check marginals
        marginals = Dict(
            X1 => [0.48, 0.52],
            X2 => [0.26, 0.74],
            X3 => [0.4, 0.6],
            X4 => [0.6, 0.4]
            )
        for (X,m) in marginals
            @test m ≈ marginal(X,bp)
        end
    end # end of Bayesian Tree testset
    @testset "Exact Inference in Markov Trees" begin
        X1 = VariableNode(Variable(1,2))
        X2 = VariableNode(Variable(2,2))
        X3 = VariableNode(Variable(3,2))
        X4 = VariableNode(Variable(4,2))
        f12 = FactorNode([10.0 0.1; 0.1 10.0], VariableNode[X1, X2]) # phi(X1,X2)
        f1 = FactorNode([5.0, 0.2], VariableNode[X1]) # phi(X1)
        f3 = FactorNode([1.0, 1.0], VariableNode[X3]) # phi(X3)
        f24 = FactorNode([5.0 0.2; 0.2 5.0], VariableNode[X2, X4]) # phi(X2,X4)
        f34 = FactorNode([0.5 20.0; 1.0 2.5], VariableNode[X3, X4]) # phi(X3,X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([X1,X2,X3,X4],[f12,f1,f3,f24,f34])
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for succificient number of iterations
        # println()
        # for ((from,to),μ) in bp.messages
        #     println(typeof(from), "->", typeof(to), ": ", μ)
        # end        
        # for i=1:2
        #     println(i, " |===============================================================")
        #     for f in fg.factors
        #         for i in eachindex(f.neighbors)
        #             update!(bp,f,i)
        #             println(f.factor, " -> ", f.neighbors[i].variable.id, " : ", bp.messages[f,f.neighbors[i]])
        #         end
        #     end
        #     println()
        #     for v in fg.variables
        #         for f in v.neighbors
        #             update!(bp,v,f)
        #             println(v.variable.id, " -> ", f.factor, " : ", bp.messages[v,f])
        #         end
        #     end
        # end
        for i=1:4
            # println(i)
            # for X in (X1,X2,X3,X4)
            #     println(X.variable, " ", marginal(X,bp))
            # end
            update!(bp)
        end   
        # check marginals
        marginals = Dict(
            X1 => [0.7440152339499456, 0.25598476605005444],
            X2 => [0.6803590859630033, 0.3196409140369967],
            X3 => [0.6521808124773303, 0.34781918752266955],
            X4 => [0.4260745375408052, 0.5739254624591947]
        )
        for (X,m) in marginals
            @test m ≈ marginal(X,bp)
        end
    end # end of Markov tree testset
    @testset "Approximate Inference in Tree-Decomposable Loopy Graph" begin
        X1 = VariableNode(Variable(1,2))
        X2 = VariableNode(Variable(2,2))
        X3 = VariableNode(Variable(3,2))
        X4 = VariableNode(Variable(4,2))
        f12 = FactorNode([10 0.1; 0.1 10], VariableNode[X1, X2]) # phi(X1,X2)
        f13 = FactorNode([5 5; 0.2 0.2], VariableNode[X1, X3]) # phi(X1,X3) = phi(X1)*phi(X3) of previous test
        f24 = FactorNode([5 0.2; 0.2 5], VariableNode[X2, X4]) # phi(X2,X4)
        f34 = FactorNode([0.5 20; 1 2.5], VariableNode[X3, X4]) # phi(X3,X4)
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph([X1,X2,X3,X4],[f12,f13,f24,f34])
        # Initialize belief progation messages
        bp = BeliefPropagation(fg)    
        # Run belief propagation for succificient number of iterations
        for i=1:4
            # println(i)
            # for X in (X1,X2,X3,X4)
            #     println(X.variable, " ", marginal(X,bp))
            # end
            update!(bp)
        end
        # check marginals
        marginals = Dict(
            X1 => [0.7440152339499456, 0.25598476605005444],
            X2 => [0.6803590859630033, 0.3196409140369967],
            X3 => [0.6521808124773303, 0.34781918752266955],
            X4 => [0.4260745375408052, 0.5739254624591947]
        )    
        for (X,m) in marginals
            @test m ≈ marginal(X,bp)
        end    
    end # end of testset Easy Loopy
end # end of testset BeliefPropagation
