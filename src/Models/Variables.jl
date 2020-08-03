# Implements data structures for random variables
export 
    Variable
    
"A Discrete Variable."
struct Variable
    id::Int
    dimension::UInt # no. of values
end
Base.show(io::IO,v::Variable) = print(io,"Variable(id=$(v.id), dim=$(v.dimension))") 
"Returns the number of values of the variable."
Base.length(v::Variable) = v.dimension