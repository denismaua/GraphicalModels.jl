using GraphicalModels
import GraphicalModels: BeliefPropagation, update!, marginal
using Test

@testset "FactorGraph" begin
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
    for v in fg.variables
        println("$(v.variable)")
        for f in v.neighbors
            println("\t", f.factor)
        end
    end
    # Run belief propagation
    bp = BeliefPropagation(fg)
   
    for i=1:4
        # for ((from,to),μ) in bp.messages
        #     println(typeof(from), "->", typeof(to), ": ", μ)
        # end
        println(i)
        for X in (X1,X2,X3,X4)
            println(X.variable, " ", marginal(X,bp))
        end
        update!(bp)

    end
    println()
    for ((from,to),μ) in bp.messages
        println(typeof(from), "->", typeof(to), ": ", μ)
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
end
