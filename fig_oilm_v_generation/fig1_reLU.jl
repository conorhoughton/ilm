#=
makes a simple agent
trains it
records the performance against generation
uses a relu hidden layer
=#

using Statistics,DataFrames,Gadfly,ProgressMeter
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

generationN=40

trialsN=25

expressMatrix=Matrix{Float64}(undef,generationN,trialsN)
composeMatrix=Matrix{Float64}(undef,generationN,trialsN)
stableMatrix =Matrix{Float64}(undef,generationN,trialsN)

progress= Progress(trialsN*generationN)

for trialC in 1:trialsN

    child = makeAgentReLU(n)
    obvert(child)
    parent=copy(child.m2s)

    for generation in 1:generationN
        
        shuffled = randperm(2^n)
        
        exemplars = shuffled[1:bottleneckN]

        expressMatrix[generation,trialC]=expressivity(child)
        composeMatrix[generation,trialC]=compositionality(child)
        
        child=makeAgent(n)

        
        totalLoss=0.0
    
        for epoch in 1:numEpochs
            
            shuffle!(exemplars)
            
            for meaning in exemplars
                dataI=[(v2BV(n,parent[meaning]-1),v2BV(n,meaning-1))]
                Flux.train!(loss, child.s2m, dataI, optimizerL)
            end
            
        
        end

        next!(progress)
        
        obvert(child)

        oldParent=copy(parent)
        parent=copy(child.m2s)
        stableMatrix[generation,trialC]=stability(parent,oldParent)    
    
    end

    
    
end


function plotProperty(propertyMatrix,xAxis,filename,color)
    mu=vec(mean(propertyMatrix, dims=2))
    stdDev = vec(std(propertyMatrix, dims=2))
    
    min = mu .- stdDev
    max = mu .+ stdDev

    df = DataFrame(x=xAxis, y=mu, yMin=min,yMax=max)

    plt=plot(layer(df,x=:x,y=:y, Geom.line),layer(df,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon),Theme(background_color=colorant"white",default_color=color))

    draw(PNG(filename, 6inch, 4inch),plt)

end

generations=collect(0:generationN-1)

plotProperty(expressMatrix,generations,"fig1_relu_express.png","blue")
plotProperty(composeMatrix,generations,"fig1_relu_compose.png","orange")
plotProperty(stableMatrix,generations,"fig1_relu_stable.png","purple")

