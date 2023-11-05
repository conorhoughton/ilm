#=
makes a simple agent
trains it
run it until it become compositional and expressive, plot the number of generations
=#

using Statistics,DataFrames,Gadfly
import Cairo, Fontconfig

include("simple_agent.jl")
include("utilities.jl")
    
n=8
bottleneckN=50

lossMSE(nn, x,y)= Flux.mse(nn(x), y)

loss(nn, x,y)= lossMSE(nn,x,y)
    
learningRateL=1.0
optimizerL=Flux.Optimise.Descent(learningRateL)

numEpochs=20

generationMax=100

trialsN=20

cutOff=0.9

bottleMin=5
bottleMax=180
bottleStep=20


bottleV=collect(bottleMin:bottleStep:bottleMax)
bottleT=length(bottleV)

waitMatrix=Matrix{Float64}(undef,bottleT,trialsN)

for bottleC in 1:bottleT

    for trialC in 1:trialsN

        global(generationMax,cutOff)
        
        child = makeAgent(n)
        obvert(child)
        parent=copy(child.m2s)
        
        express=0.0
        compose=0.0

        generation=1
        
        while ((generation <= generationMax) && (express<cutOff || compose<cutOff))

            shuffled = randperm(2^n)
            
            exemplars = shuffled[1:bottleV[bottleC]]
            
            child=makeAgent(n)
                        
            totalLoss=0.0
            
            for epoch in 1:numEpochs
                
                shuffle!(exemplars)
                
                for meaning in exemplars
                    dataI=[(v2BV(n,parent[meaning]-1),v2BV(n,meaning-1))]
                    Flux.train!(loss, child.s2m, dataI, optimizerL)
                end
                        
            end
            
            obvert(child)
            
            parent=copy(child.m2s)
            
            express=expressivity(child)
            compose=compositionality(child)

            generation+=1

#=            println(generation," ",bottleNeckC," ",express," ",compose," ",cutOff," ",
            (generation <= generationMax) && (express<cutOff || compose<cutOff))
=#
        end

        waitMatrix[bottleC,trialC]=generation
        
    end
    
end
        
mu=vec(mean(waitMatrix, dims=2))
stdDev = vec(std(waitMatrix, dims=2))

min = mu .- stdDev
max = mu .+ stdDev

expressDf = DataFrame(X=bottleV, Y=mu, StdDev=stdDev,YMin=min,YMax=max)

plt=plot(layer(expressDf,x=:X,y=:Y, Geom.line),layer(expressDf,x=:X,ymin=:YMin,ymax=:YMax,Geom.ribbon),Theme(background_color=colorant"white",default_color="red"))

draw(PNG("fig2.png", 6inch, 4inch),plt)

