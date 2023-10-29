
include("simple_agent.jl")
include("utilities.jl")
    
n=8
bottleneckN=50

lossMSE(nn, x,y)= Flux.mse(nn(x), y)

function lossXE(nn,x,y)
    epsilon=0.00001
    loss=0.0
    xHat=nn(x)
    for i in 1:length(x)
        loss-=xHat[i]*log2(y[i]+epsilon)+(1-xHat[i])*log2(1-y[i]+epsilon)
    end
    loss
end

loss(nn, x,y)= lossMSE(nn,x,y)

learningRate=1.0
optimizer=Flux.Optimise.Descent(learningRate)

numEpochs=20

generationN=40

parent=makeAgent(n)
obvert(parent)
parentM2s=copy(parent.m2s)

for generation in 1:generationN
    global(parentM2s)

    child=makeAgent(n)
    exemplars = randperm(2^n)[1:bottleneckN]

    lossXETotal=0.0
    lossMSETotal=0.0
    
    for epoch in 1:numEpochs
        shuffle!(exemplars)
        for meaning in exemplars
            dataI=[(v2BV(n,parentM2s[meaning]-1),v2BV(n,meaning-1))]
            Flux.train!(loss, child.s2m, dataI, optimizer)
            lossXETotal+=lossXE(child.s2m,dataI[1][1],dataI[1][2])
            lossMSETotal+=lossMSE(child.s2m,dataI[1][1],dataI[1][2])
        end
    end
    
    obvert(child)
    println(generation," ",expressivity(child)," ",compositionality(child)," ",lossMSETotal," ",lossXETotal)
    parentM2s=copy(child.m2s)
end
        
