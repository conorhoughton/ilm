
include("simple_agent.jl")
include("utilities.jl")
    
n=8
bottleneckN=50

loss(nn, x,y)= Flux.mse(nn(x), y)

learningRate=1.0

optimizer=Flux.Optimise.Descent(learningRate)

numEpochs=20

generationN=50

parent=makeAgent(n)
obvert(parent)
parentM2s=copy(parent.m2s)

for generation in 1:generationN
    global(parentM2s)

    child=makeAgent(n)
    exemplars = randperm(2^n)[1:bottleneckN]

    for epoch in 1:numEpochs
        shuffle!(exemplars)
        for meaning in exemplars
            dataI=[(v2BV(n,parentM2s[meaning]-1),v2BV(n,meaning-1))]
            Flux.train!(loss, child.s2m, dataI, optimizer)
        end
    end
    
    obvert(child)
    #println(generation," ",expressivity(child))
    println(generation," ",compositionality(child))
    parentM2s=copy(child.m2s)
    
end
        
