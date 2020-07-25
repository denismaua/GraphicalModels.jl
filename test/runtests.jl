using GraphicalModels
using Test

@testset "FactorGraph" begin
    X4 = VariableNode(4)
    X3 = VariableNode(3)
    X2 = VariableNode(2)
    X1 = VariableNode(1)
    f1 = FactorNode([0.3 0.6; 0.7 0.4], VariableNode[X1, X3]) # P(X1|X3)
    f2 = FactorNode([0.5 0.1; 0.5 0.9], VariableNode[X2, X3]) # P(X2|X3)
    f3 = FactorNode([0.2 0.7; 0.8 0.3], VariableNode[X3, X4]) # P(X3|X4)
    f4 = FactorNode([0.6, 0.4], [X4]) # P(X4)
    fg = FactorGraph([X1,X2,X3,X4],[f1,f2,f3,f4])
    # push!(X4.neighbors,f4)
    # push!(X4.neighbors,f3)
    # push!(X3.neighbors,f3)
    # push!(X3.neighbors,f2)
    # push!(X3.neighbors,f1)
    # push!(X1.neighbors,f1)
    # push!(X2.neighbors,f2)
    for v in fg.variables
        println("$(v.variable)")
        for f in v.neighbors
            println("\t", f.factor)
        end
    end
end
