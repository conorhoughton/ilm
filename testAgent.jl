include("simple_agent.jl")
include("utilities.jl")

function totalLoss(nn,training)
    totalLoss=0.0
    for i in 1:length(training[1])
        dataI=[training[1][i],training[2][i]]
        totalLoss+=loss(nn, dataI[1],dataI[2])
    end
    totalLoss
end
    

 
n=8

agent=makeAgent(n)

sampleN=50

training=makeXY(n,sampleN)

loss(nn, x,y)= Flux.mse(nn(x), y)

learning_rate=10

optimizer=Flux.Optimise.Descent(learning_rate)

numEpochs=100



for epoch in 1:numEpochs
    randomOrder=shuffle(1:sampleN)
    for i in randomOrder
        dataI=[(training[1][i],training[2][i])]
        Flux.train!(loss, agent.s2m, dataI, optimizer)
    end
    println(totalLoss(agent.s2m,training))
end

obvert(agent)

println(expressivity(agent))
