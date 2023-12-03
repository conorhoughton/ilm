#=
makes a simple agent
trains it
records the performance against generation
=#

using Statistics,DataFrames,Gadfly,ProgressMeter,Colors
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

generationN=35

trialsN=5

expressMatrix=Matrix{Float64}(undef,generationN,trialsN)
composeMatrix=Matrix{Float64}(undef,generationN,trialsN)
stableMatrix =Matrix{Float64}(undef,generationN,trialsN)


bgCompose=0.0::Float64
bgExpress=0.0::Float64
bgStable =0.0::Float64
backgroundN=20

for backgroundC in 1:backgroundN

    global(bgCompose,bgExpress,bgStable)
    child=makeAgent(n)
    anotherChild=makeAgent(n)
    obvert(child)
    bgCompose+=newCompose(child)/backgroundN
    bgExpress+=expressivity(child)/backgroundN
    bgStable+=stability(child.m2s,anotherChild.m2s)/backgroundN
    
end

rebased(value,bg)=(value-bg)/(1.0-bg)

println("background")
println("compose"," ",bgCompose)
println("express"," ",bgExpress)
println("stable "," ",bgStable)

progress= Progress(trialsN*generationN)

for trialC in 1:trialsN

    child = makeAgent(n)
    obvert(child)
    parent=copy(child.m2s)

    global(bgCompose,bgExpress,bgStable)

    for generation in 1:generationN
        
        shuffled = randperm(2^n)
        
        exemplars = shuffled[1:bottleneckN]

        expressMatrix[generation,trialC]=rebased(expressivity(child),bgExpress)
        composeMatrix[generation,trialC]=rebased(newCompose(child),bgCompose)
        
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
        stableMatrix[generation,trialC]=rebased(stability(parent,oldParent),bgStable)    
    
    end

    
    
end


function plotProperty(propertyMatrix,xAxis,filename,color)
    mu=vec(mean(propertyMatrix, dims=2))
    stdDev = vec(std(propertyMatrix, dims=2))
    
    min = mu .- stdDev
    max = mu .+ stdDev

    df = DataFrame(x=xAxis, y=mu, yMin=min,yMax=max)

    plt=plot(
        layer(df,x=:x,y=:y, Geom.line,style(line_width=3pt)),
        layer(df,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon),
        Theme(background_color=colorant"white",default_color=color))

    draw(PNG(filename, 2.5inch, 2inch),plt)

end


function plotPropertyLines(propertyMatrix,filename,color,yLabel)

    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    df=df[(df.time .!= 1) , :]
    
    avgPerformance = mean(propertyMatrix, dims=2)[2:end] |> vec
    avgDf = DataFrame(time=2:numTimePoints, avgPerformance=avgPerformance)

    
    
    alphaValue=0.3
    lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial, Geom.line,style(line_width=0.5pt,default_color=lightColor)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0)
     )

    draw(PNG(filename, 2.5inch, 2inch),plt)
    
end

plotPropertyLines(expressMatrix,"oilm_express.png",colorant"blue","e")
plotPropertyLines(composeMatrix,"oilm_compose.png",colorant"orange","c")
plotPropertyLines(stableMatrix,"oilm_stable.png" ,colorant"purple","s")

generations=collect(0:generationN-1)

plotProperty(expressMatrix,generations,"fig1_express.png","blue")
plotProperty(composeMatrix,generations,"fig1_compose.png","orange")
plotProperty(stableMatrix,generations,"fig1_stable.png","purple")

