# Implements data structures for random variables

"A Discrete Variable."
struct Variable
    id::Int
    dimension::UInt # no. of values
end
Base.show(io::IO,v::Variable) = print(io,"Variable(id=$(v.id), dim=$(v.dimension))") 
"Returns the number of values of the variable."
Base.length(v::Variable) = v.dimension

Base.hash(v::Variable, h::UInt) = hash(v.id, h)
Base.isequal(x::Variable,y::Variable) = isequal(x.id,y.id)