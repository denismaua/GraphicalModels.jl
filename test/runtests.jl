using GraphicalModels
import GraphicalModels: VariableNode, FactorNode, FactorGraph
using Test

@testset "Inference" begin
    include("testspbp.jl")
    include("testhbp.jl")
end

@testset "I/O" begin
    include("testloadfg.jl")
end

