# Test loading of factor graphs from data
@testset "Loading factor graphs from file" begin
    @testset "Loading 3-variable factor graph from file" begin
        import GraphicalModels: BeliefPropagation, update!, marginal

        # load simple model described in https://www.cs.huji.ac.il/project/UAI10/fileFormat.php
        fg = FactorGraph(normpath("$(@__DIR__)/markov.uai"))
        # now test if marginals are correctly computed by belief propagation
        bp = BeliefPropagation(fg)    
        while update!(bp) > 1e-10 && bp.iterations < 10 end
        @info "converged in $(bp.iterations) iterations."
        marginals = Dict(
            "0" => [0.436, 0.564],
            "1" => [0.574688, 0.425312],
            "2" => [0.465612512, 0.191371104, 0.343016384]
        )
        @testset "Checking marginal for $i" for (i,x) in fg.variables
            @test marginals[i] ≈ marginal(x,bp)
        end


        # fg = FactorGraph(normpath("$(@__DIR__)/example.uai"))
    end
    @testset "Loading more complex model from file" begin
        import GraphicalModels: BeliefPropagation, update!, marginal, setevidence!
        # load cyclic model
        fg = FactorGraph(normpath("$(@__DIR__)/example2.uai"))  
        @info length(fg.variables), length(fg.factors)
        bp = BeliefPropagation(fg)    
        setevidence!(bp,"0",2)
        setevidence!(bp,"1",2)
        while update!(bp) > 1e-10 && bp.iterations < 10 
            @info marginal("16",bp)
        end
        @info "finished in $(bp.iterations) iterations."

    end
end