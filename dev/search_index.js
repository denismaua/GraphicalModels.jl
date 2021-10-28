var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GraphicalModels","category":"page"},{"location":"#GraphicalModels","page":"Home","title":"GraphicalModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GraphicalModels]","category":"page"},{"location":"#GraphicalModels.FGNode","page":"Home","title":"GraphicalModels.FGNode","text":"A node of a Factor Graph.\n\n\n\n\n\n","category":"type"},{"location":"#GraphicalModels.FactorGraph","page":"Home","title":"GraphicalModels.FactorGraph","text":"A Factor Graph is a bipartite graph where nodes are either variables or factors.\n\nArguments\n\nvariables: vector of variable nodes.\nfactors: vector of factor nodes.\n\n\n\n\n\n","category":"type"},{"location":"#GraphicalModels.FactorGraph-Tuple{String}","page":"Home","title":"GraphicalModels.FactorGraph","text":"FactorGraph(filename::AbstractString)::FactorGraph\nFactorGraph(io::IO=stdin)::FactorGraph\n\nReads a model from file in UAI Competition Format and returns the correspoding factor graph. See https://www.cs.huji.ac.il/project/UAI10/fileFormat.php for details about the file format.\n\n\n\n\n\n","category":"method"},{"location":"#GraphicalModels.FactorNode","page":"Home","title":"GraphicalModels.FactorNode","text":"Representes a factor node.\n\nArguments\n\nfactor: a multidimensional array representing a function of the neighbors. Each dimension corresponds to the order of the variable in the vector neighbors\nneighbors: adjacent variable nodes\n\n\n\n\n\n","category":"type"},{"location":"#GraphicalModels.Variable","page":"Home","title":"GraphicalModels.Variable","text":"A Discrete Variable.\n\n\n\n\n\n","category":"type"},{"location":"#GraphicalModels.VariableNode","page":"Home","title":"GraphicalModels.VariableNode","text":"Represents a variable node.\n\nArguments\n\nvariable: integer identifying variable\nneighbors: adjacent factor nodes\n\n\n\n\n\n","category":"type"},{"location":"#Base.length-Tuple{GraphicalModels.Variable}","page":"Home","title":"Base.length","text":"Returns the number of values of the variable.\n\n\n\n\n\n","category":"method"}]
}