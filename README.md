# GraphicalModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://denismaua.github.io/GraphicalModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://denismaua.github.io/GraphicalModels.jl/dev)
[![Build Status](https://github.com/denismaua/GraphicalModels.jl/workflows/CI/badge.svg)](https://github.com/denismaua/GraphicalModels.jl/actions)

This package is a lightweight implementation of Probabilistic Graphical Model algorithms in [Julia](https://julialang.org), built for didatic purposes.

## Features

Currently, it handles manipulation of Discrete Factor Graphs (build using the API or by loading from file in the UAI Competition format), and approximate inference through belief propagation (marginal, max-marginal and mixed-marginal inferences).

A Factor Graph is a bipartite graph consisting of Variable nodes and Factor nodes. Variable nodes are associated with random variables and Factor nodes are associated with functions whose domain is the direct product of the neighboring (variable) nodes. In the simplest discrete case, a factor node is associated with a multidimensional array (a factor) representing the function.

## Instalation

```julia
import Pkg
Pkg.add("https://github.com/denismaua/GraphicalModels.jl")
```

## Usage

Specifying a Factor Graph as a pair composed of a dictionary Name => VariableNode and a dictionary Name => FactorNode:

```julia
import GraphicalModels: VariableNode, FactorNode, FactorGraph
x = VariableNode(2) # a binary variable
y = VariableNode(2) # a binary variable
z = VariableNode(3) # a ternary variable
# A factor representing an unconditional distribution P(x)
f = FactorNode(log.([0.436, 0.564]), [x]) # values should be in log domain
# A factor representing a conditional distibution P(y|x)
g = FactorNode(log.([0.128 0.872; 0.920 0.080]), [x,y]) # factors are internally represented as multidimensional arrays whose dimensions are the variables in their scope (in the given ordering)
# A factor representinf a conditional distribution P(z|y)
h = FactorNode(log.([0.210 0.333 0.457; 0.811 0.000 0.189 ]), [y,z])
# Now create the corresponding factor graph (connects variable nodes and factor nodes) - Note: the names/labels of nodes are arbitrary strings
fg = FactorGraph(
    Dict("X" => x, "Y" => y, "Z" => z),             # Variable Nodes
    Dict("P(X)" => f, "P(Y|X)" => g, "P(Z|Y)" => h) # Factor Nodes
)
# to show the variables in the factor graph
foreach(println, fg.variables) # show variables
````

Specifying a Factor Graph as a pair List of VariableNodes and List of FactorNodes:

```julia
import GraphicalModels: VariableNode, FactorNode, FactorGraph
x = VariableNode(2) # binary variable
y = VariableNode(2) # binary variable
z = VariableNode(3) # ternary variable
# Specify P(x)
f = FactorNode(log.([0.436, 0.564]), [x]) # values should be in log domain
# Specify P(y|x)
g = FactorNode(log.([0.128 0.872; 0.920 0.080]), [x,y]) 
# Specify P(z|y)
h = FactorNode(log.([0.210 0.333 0.457; 0.811 0.000 0.189 ]), [y,z])
# Now create factor graph; nodes will be named after their order ("1", "2", ...)
fg = FactorGraph(
    [x,y,z],
    [f,g,h]
)
foreach(println, fg.variables) # show variables in the graph
````

Loading factor graph from file in the [UAI 2010 Competition File Format](https://www.cs.huji.ac.il/project/UAI10/fileFormat.php)):

```julia
# Load simple example in https://www.cs.huji.ac.il/project/UAI10/fileFormat.php
fg = FactorGraph("test/markov.uai")
foreach(println, fg.variables) # show variables
```

Computing marginals by sum-product belief propagation:

```julia
import GraphicalModels.MessagePassing: BeliefPropagation, update!, marginal
# Load model from file
fg = FactorGraph("test/markov.uai")
# Initialize belief progation messages
bp = BeliefPropagation(fg)
# Run belief propagation for until convergence
while true
    res = update!(bp) # update all messages, returns maximum change (residual)
    @info "Residual: $res"
    if res < 1e-6  # convergence tolerance
        break
    end
end
@info "Converged in $(bp.iterations) iterations."
# Now compute marginal for variable Y
@show marginal(bp, "1")
# Alternatively, we can use marginal(bp, fg.variables["1"])
```

Computing conditional marginal probabilities:

```julia
import GraphicalModels.MessagePassing: BeliefPropagation, update!, marginal, setevidence!
# Load model from file
fg = FactorGraph("test/markov.uai")
# Initialize belief progation messages
bp = BeliefPropagation(fg)
# Set evidence Z = 2
setevidence!(bp, "2", 2)
# Run belief propagation for until convergence
while update!(bp) > 1e-10 end
@info "converged in $(bp.iterations) iterations."
# Now compute marginal for variable X
@show marginal(bp, "0")
```
