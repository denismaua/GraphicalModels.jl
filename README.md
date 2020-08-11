# GraphicalModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://denismaua.github.io/GraphicalModels.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://denismaua.github.io/GraphicalModels.jl/dev)
[![Build Status](https://github.com/denismaua/GraphicalModels.jl/workflows/CI/badge.svg)](https://github.com/denismaua/GraphicalModels.jl/actions)

This package is a lightweight implementation of Probabilistic Graphical Models algorithms, built for didatical purposes.

## Instalation

```julia
import Pkg
Pkg.add("https://github.com/denismaua/GraphicalModels.jl")
```

## Usage

Specifying a Factor Graph.

```julia
import GraphicalModels: VariableNode, FactorNode, FactorGraph
x = VariableNode(2) # binary variable X
y = VariableNode(2) # binary variable Y
z = VariableNode(2) # ternary variable Z
# Specify P(X)
f = FactorNode(log.([0.436 0.564]), [x]) # values should be in log domain
# Specify P(Y|X)
g = FactorNode(log.([0.128 0.872; 0.920 0.080]), [x,y]) # factors are multidimensional arrays whose dimensions are given by the dimensions of the variables in their scope (in the given ordering)
# Specify P(Z|Y)
h = FactorNode(log.([0.210 0.333 0.457; 0.811 0.000 0.189 ]), [y,z])
# Now create factor graph (this adds neighbor links to variable nodes)
fg = FactorGraph(
    Dict("X" => x, "Y" => y, "Z" => z),
    Dict("P(X)" => f, "P(Y|X)" => g, "P(Z|Y)" => h)
)
````

For automatically specifying indices from positions:

```julia
import GraphicalModels: VariableNode, FactorNode, FactorGraph
x = VariableNode(2) # binary variable X
y = VariableNode(2) # binary variable Y
z = VariableNode(2) # ternary variable Z
# Specify P(X)
f = FactorNode(log.([0.436 0.564]), [x]) # values should be in log domain
# Specify P(Y|X)
g = FactorNode(log.([0.128 0.872; 0.920 0.080]), [x,y]) # factors are multidimensional arrays whose dimensions are given by the dimensions of the variables in their scope (in the given ordering)
# Specify P(Z|Y)
h = FactorNode(log.([0.210 0.333 0.457; 0.811 0.000 0.189 ]), [y,z])
# Now create factor graph (this adds neighbor links to variable nodes)
fg = FactorGraph(
    [x,y,z],
    [f,g,h]
)
foreach(println, fg.variables) # show variables
````

Loading factor graph from file (in UAI File Format):

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
while True
    res = update!(bp) # update all messages, returns maximum change (residual)
    @info "Residual: $res"
    if res < 1e-6  # convergence tolerance
        break
    end
end
@info "Converged in $(bp.iterations) iterations."
# Now compute marginal for variable Y
@show marginal("1", bp)
# Alternatively, we can use marginal(fg.variables["1"], bp)
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
@show marginal("0", bp)
```
