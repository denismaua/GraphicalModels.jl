@testset "MaxSumBeliefPropagation" begin
    import GraphicalModels.MessagePassing: MaxSum, update!, marginal, decode
    @testset "MaxSum Inference in Bayes Tree" begin
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
        # Initialize max-sum belief propagation algorithm
        bp = MaxSum(fg)
        messages = [(x1,f1), (x2,f2), (f1,x3), (f2,x3), (x3,f3), (f3,x4), (f4,x4), (x4,f3), (f3,x3), (x3,f1), (x3,f2), (f1,x1), (f2,x2)]
        for (f,t) in messages
            update!(bp,f,t)
            # @show exp.(bp[f,t])
        end
        # Run message passing until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
        #@test bp.iterations < 3
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
    @testset "MaxSum in Chain Graph" begin
        # Lemniscate topology graph
        x1 = VariableNode(2)
        x2 = VariableNode(2)
        x3 = VariableNode(2)
        f12 = FactorNode(-2*[50 60; 20 300], VariableNode[x1, x2])
        f23 = FactorNode(-2*[45 100; 300 6], VariableNode[x2, x3]) 
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph(
            Dict( "X1" => x1, "X2" => x2, "X3" => x3 ),
            Dict( "F12" => f12, "F23" => f23 )
        )
        # Initialize max-sum belief propagation algorithm
        bp = MaxSum(fg)
        # Run message passing until convergence
        while update!(bp) > 1e-10 end
        @info "converged in $(bp.iterations) iterations."
    #     @test bp.iterations < 3
    #     # check marginals    
        map = Dict(
            x1 => 2,
            x2 => 1,
            x3 => 1
            )    
        # for (i,x) in fg.variables
        #     @info marginal(bp, x)
        #     @info "$i = $(decode(bp, x))"
        # end
        @testset "Checking decoding for $i" for (i,x) in fg.variables
            @test map[x] == decode(bp,x)
        end
    end      
    # @testset "MaxSum in Loopy Graph" begin
    #     # Lemniscate topology graph - Example where maxsum does not converge
    #     x1 = VariableNode(2)
    #     x2 = VariableNode(2)
    #     x3 = VariableNode(2)
    #     f12 = FactorNode(-[50 60; 20 0], VariableNode[x1, x2])
    #     f21 = FactorNode(-[50 20; 60 600], VariableNode[x2, x1]) 
    #     # f23 = FactorNode(-[45 100; 300 6], VariableNode[x2, x3]) 
    #     # f32 = FactorNode(-[45 300; 100 6], VariableNode[x3, x2]) 
    #     f23 = FactorNode(-[45 150; 0 6], VariableNode[x2, x3]) 
    #     f32 = FactorNode(-[45 600; 50 6], VariableNode[x3, x2]) 
    #     # f23 = FactorNode(-[100 70; 50 20], VariableNode[x2, x3]) 
    #     # f32 = FactorNode(-[100 50; 70 20], VariableNode[x3, x2]) 
    #     # Creates factor graph (adds neighbor links to variables)
    #     fg = FactorGraph(
    #         Dict( "X1" => x1, "X2" => x2, "X3" => x3 ),
    #         Dict( "F12" => f12, "F21" => f21, "F23" => f23, "F32" => f32 )
    #     )
    #     # Initialize max-sum belief propagation algorithm
    #     bp = MaxSum(fg)
    #     #bp.normalize = true
    #     #bp.λ = 0.999
    #     # Run message passing until convergence
    #     # messages = [
    #     #     (x1,f12), # 1
    #     #     (x1,f21), # 1
    #     #     (f12,x2), # 2
    #     #     (f21,x2), # 2
    #     #     (x2,f32), # 3
    #     #     (x2,f23), # 3
    #     #     (f32,x3), # 4
    #     #     (f23,x3), # 4
    #     #     (x3,f23), # 5
    #     #     (x3,f32), # 5
    #     #     (f23,x2), # 6
    #     #     (f32,x2), # 6
    #     #     (x2,f21), # 7
    #     #     (x2,f12), # 7
    #     #     (f21,x1),
    #     #     (f12,x1)
    #     # ]
    #     messages = [
    #         (x1, f12), (x1, f21),
    #         (f12, x1), (f12, x2),
    #         (x2, f12), (x2, f21), (x2, f23), (x2, f32),
    #         (f21, x1), (f21, x2),
    #         (f23, x2), (f23, x3),
    #         (x3, f23), (x3, f32),
    #         (f32, x2), (f32, x3)
    #     ]
    #     # update!(bp, f21, x1)
    #     # update!(bp, f12, x1)
    #     # @info bp.messages[f12, x1]
    #     # @info bp.messages[f21, x1]
    #     # for i = 1:100
    #     #     @info "iteration $i"    
    #     #     res = 0.0        
    #     #     for (f,t) in messages
    #     #         res = max(res,update!(bp,f,t))            
    #     #         # @show exp.(bp[f,t])
    #     #     end     
    #     #     @info "residual: $res"   
    #     #     for (i,x) in fg.variables
    #     #         @info "$i = $(decode(bp, x))"
    #     #     end
    #     # end
    #     last_occ = zeros(Int64, 8)
    #     period = zeros(Int64, 8)
    #     T = 100000
    #     for t=1:T
    #         #res = update!(bp)
    #         res = 0.0        
    #         for (from,to) in messages
    #             res = max(res,update!(bp,from,to))            
    #         end     
    #         #@info "residual: $res"   
    #         if t > T-1000
    #             X1 = decode(bp, x1)
    #             X2 = decode(bp, x2)
    #             X3 = decode(bp, x3)
    #             s = 1+(X1-1) + 2*(X2-1) + 4*(X3-1)
    #             if last_occ[s] > T-1000
    #                 period[s] = t - last_occ[s]
    #             end
    #             last_occ[s] = t
    #             # @info "iteration $i | $res"            
    #             # for (i,x) in fg.variables
    #             #     @info "$i = $(decode(bp, x))"
    #             # end
    #             if t % 150 == 0
    #                 println("$t $X1 $X2 $X3 $period")
    #             end
    #         end
    #     end
    #     X1 = decode(bp, x1)
    #     X2 = decode(bp, x2)
    #     X3 = decode(bp, x3)
    #     s = 1+(X1-1) + 2*(X2-1) + 4*(X3-1)
    #     println("$T $X1 $X2 $X3 $period")
    #     @info "done"
    #     # while update!(bp) > 1e-10 
    #     #     @info "it $(bp.iterations)"
    #     #     for (i,x) in fg.variables
    #     #         @info "$i = $(decode(bp, x))"
    #     #     end
    #     # end        
    #     #@info "converged in $(bp.iterations) iterations."
    # #     @test bp.iterations < 3
    # #     # check marginals    
    # #     optimum = Dict(
    # #         x1 => 2,
    # #         x2 => 1,
    # #         x3 => 1
    # #         )    
    #     for (i,x) in fg.variables
    #         @info marginal(bp, x)
    #         @info "$i = $(decode(bp, x))"
    #     end
    # #     @testset "Checking marginal for $i" for (i,x) in fg.variables
    # #         @test marginals[x] ≈ marginal(bp,x)
    # #     end
    # end
    @testset "4-Node XOR Grid" begin
        # Example where MaxSum Marginals have quasi-ties and damped max-sum converges to optimum
        a = VariableNode(2)
        b = VariableNode(2)
        c = VariableNode(2)
        d = VariableNode(2)
        fA = FactorNode([0.1, 0], VariableNode[a]) # add small value to ensure uniqueness of optimum
        # fB = FactorNode([0., 0.0], VariableNode[b])
        # fC = FactorNode([0., 0.0], VariableNode[c])
        # fD = FactorNode([0.0, 0.], VariableNode[d])

        # fAC = FactorNode([5.0 0.2; 5.0 0.2], VariableNode[a, c])
        # fAB = FactorNode([10. 0.1; 0.1 10.], VariableNode[a, b]) 
        # fBD = FactorNode([5. 0.2; 0.2 5.], VariableNode[b, d]) 
        # fCD = FactorNode([0.5 20.; 1.0 2.5], VariableNode[c, d]) 
        fAC = FactorNode([-1. 1.; 1. -1.], VariableNode[a, c])
        fAB = FactorNode([-1. 1.; 1. -1.], VariableNode[a, b]) 
        fBD = FactorNode([-1. 1.; 1. -1.], VariableNode[b, d]) 
        fCD = FactorNode([-1. 1.; 1. -1.], VariableNode[c, d]) 
        fAD = FactorNode([-1. 1.; 1. -1.], VariableNode[a, d]) 
        # Creates factor graph (adds neighbor links to variables)
        fg = FactorGraph(
            Dict( "A" => a, "B" => b, "C" => c, "D" => d ),
            Dict(
                "FA" => fA, 
                #"FB" => fB, "FC" => fC, "FD" => fD, 
                "FAB" => fAB, 
                "FAD" => fAD,
                "FAC" => fAC, 
                "FBD" => fBD, 
                "FCD" => fCD 
            )
        )
        # Initialize max-sum belief propagation algorithm
        bp = MaxSum(fg)
        bp.λ = 0.5
        update!(bp, fA, a)
        # Run message passing until convergence
        while update!(bp) > 1e-10 && bp.iterations < 1000 end
        @info "converged in $(bp.iterations) iterations."
        @test bp.iterations < 1000
        # map = Dict(
        #     a => 1,
        #     b => 2,
        #     c => 2,
        #     d => 1
        #     )    
        # @testset "Checking decoding for $i" for (i,x) in fg.variables
        #     @info "$i = $(decode(bp, x)) | $(marginal(bp,x))"
        #     #@test map[x] == decode(bp,x)
        # end
    end      
end