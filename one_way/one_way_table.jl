
include("one_way_agent.jl")
include("utilities.jl")
    
n=8
bottleneckN=50
constrastN=0

lossMSE(nn, x, y)= Flux.mse(nn(x), y)

loss(nn, x,y)= lossMSE(nn,x,y)

constrast(nn, x1, x2) = Flux.mse(nn(x1),nn(x2))

learningRateL=1.0
optimizerL=Flux.Optimise.Descent(learningRateL)

numEpochs=20

generationN=40

parent=makeAgent(n)

for generation in 1:generationN

    global(parent)

    child=makeAgent(n)
    
    exemplars = randperm(2^n)[1:bottleneckN]

    totalLoss=0.0
    
    for epoch in 1:numEpochs
        
        shuffle!(exemplars)

        for meaning in exemplars
            dataI=[(v2BV(n,meaning-1),round.(parent.m2s(v2BV(n,meaning-1))))]
            Flux.train!(loss, child.m2s, dataI, optimizerL)
            if epoch==numEpochs
                totalLoss+=loss(child.m2s,dataI[1][1],dataI[1][2])
            end
        end
                
    end
    
    println(generation," ",expressivity(child)," ",totalLoss)
    parent=deepcopy(child)
    
end
        
