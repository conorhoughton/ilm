#=
makes a simple agent
trains it
records the performance against bottleneck
=#

using Statistics,DataFrames,Gadfly,ProgressMeter,Colors
import Cairo, Fontconfig

include("simple_agent.jl")
include("utilities.jl")
    
n=8
#bottleneckN=50
bottle0=10
bottle1=150
bottleStep=10

bottleSize=length(collect(bottle0:bottleStep:bottle1))


lossMSE(nn, x,y)= Flux.mse(nn(x), y)

loss(nn, x,y)= lossMSE(nn,x,y)
    
learningRateL=1.0
optimizerL=Flux.Optimise.Descent(learningRateL)

numEpochs=20*50

generation0=20
generation1=50

trialsN=25

expressMatrix0=Matrix{Float64}(undef,bottleSize,trialsN)
composeMatrix0=Matrix{Float64}(undef,bottleSize,trialsN)
stableMatrix0 =Matrix{Float64}(undef,bottleSize,trialsN)


expressMatrix1=Matrix{Float64}(undef,bottleSize,trialsN)
composeMatrix1=Matrix{Float64}(undef,bottleSize,trialsN)
stableMatrix1 =Matrix{Float64}(undef,bottleSize,trialsN)



bgCompose=0.0::Float64
bgExpress=0.0::Float64
bgStable =0.0::Float64
backgroundN=20

for backgroundC in 1:backgroundN

    global(bgCompose,bgExpress,bgStable)
    child=makeAgent(n)
    obvert(child)
    GC.gc()
    anotherChild=makeAgent(n)
    obvert(anotherChild)
    GC.gc()
    bgCompose+=0.5*(compositionality(child)+compositionality(anotherChild))/backgroundN
    bgExpress+=0.5*(expressivity(child)+expressivity(anotherChild))/backgroundN
    bgStable+=stability(child.m2s,anotherChild.m2s)/backgroundN
    
end

println("background")
println("compose"," ",bgCompose)
println("express"," ",bgExpress)
println("stable "," ",bgStable)

progress= Progress(trialsN*bottleSize*generation1)

for trialC in 1:trialsN
    
    global(bgCompose,bgExpress,bgStable)

    for (bottleIndex,bottleC) in enumerate(collect(bottle0:bottleStep:bottle1))

        child = makeAgent(n)
        obvert(child)
        parent=copy(child.m2s)

        for generation in 1:generation1
        
            shuffled = randperm(2^n)
        
            exemplars = shuffled[1:bottleC]

            if generation==generation0
                expressMatrix0[bottleIndex,trialC]=rebased(expressivity(child),bgExpress)
                composeMatrix0[bottleIndex,trialC]=rebased(compositionality(child),bgCompose)
            elseif generation==generation1
                expressMatrix1[bottleIndex,trialC]=rebased(expressivity(child),bgExpress)
                composeMatrix1[bottleIndex,trialC]=rebased(compositionality(child),bgCompose)
            end
        
            child=makeAgent(n)

            totalLoss=0.0

            normalizedEpochs=round(Int64,numEpochs/bottleC)
            
            for epoch in 1:normalizedEpochs
                
                shuffle!(exemplars)
                
                for meaning in exemplars
                    dataI=[(v2BV(n,parent[meaning]-1),v2BV(n,meaning-1))]
                    Flux.train!(loss, child.s2m, dataI, optimizerL)
                end
                
                
            end
            
            next!(progress)
        
            obvert(child)
            GC.gc()
            
            oldParent=copy(parent)
            parent=copy(child.m2s)
            if generation==generation0
                stableMatrix0[bottleIndex,trialC]=rebased(stability(parent,oldParent),bgStable)    
            elseif generation==generation1
                stableMatrix1[bottleIndex,trialC]=rebased(stability(parent,oldParent),bgStable)    
            end
        end
    end
    
    
end


function plotProperty(propertyMatrix,xAxis,filename,color,yLabel)
    mu=vec(mean(propertyMatrix, dims=2))
    stdDev = vec(std(propertyMatrix, dims=2))
    
    min = mu .- stdDev
    max = mu .+ stdDev

    df = DataFrame(x=xAxis, y=mu, yMin=min,yMax=max)
       
    #alphaValue=0.3
    #lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(df,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon,style(default_color=color)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("bottleneck"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0,ymax=1.0)

    )

    draw(PNG(filename, 2.5inch, 2inch),plt)

end


function plotProperty(propertyMatrix0,propertyMatrix1,xAxis,filename,color,yLabel)


    function muMinMax(propertyMatrix)
    
        mu=vec(mean(propertyMatrix, dims=2))
        stdDev = vec(std(propertyMatrix, dims=2))
    
        (mu,mu .- stdDev,mu .+ stdDev)
    end

    (mu0,min0,max0)=muMinMax(propertyMatrix0)
    (mu1,min1,max1)=muMinMax(propertyMatrix1)
    
    df0 = DataFrame(x=xAxis, y=mu0, yMin=min0,yMax=max0)
    df1 = DataFrame(x=xAxis, y=mu1, yMin=min1,yMax=max1)
       
    #alphaValue=0.3
    #lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color,line_style=[:dash])),
        layer(df1,x=:x,y=:y, Geom.line,style(line_width=1pt,default_color=colorant"black",line_style=[:dash])),
        layer(df0,x=:x,y=:y, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df0,x=:x,ymin=:yMin,ymax=:yMax,Geom.ribbon,style(default_color=color)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("bottleneck"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0,ymax=1.0)

    )

    draw(PNG(filename, 2.5inch, 2inch),plt)

end

    

function plotPropertyLines(propertyMatrix,filename,color,yLabel)

    numTimePoints = size(propertyMatrix, 1)
    numTrials = size(propertyMatrix, 2)

    df = DataFrame()
    df.time = repeat(1:numTimePoints, outer=numTrials)
    df.trial = repeat(1:numTrials, inner=numTimePoints)
    df.performance = vec(propertyMatrix)

    avgPerformance = mean(propertyMatrix, dims=2)[1:end] |> vec
    avgDf = DataFrame(time=1:numTimePoints, avgPerformance=avgPerformance)

       
    alphaValue=0.3
    lightColor = RGBA(color, alphaValue)
    
    plt=plot(
        layer(avgDf, x=:time, y=:avgPerformance, Geom.line,style(line_width=3pt,default_color=color)),
        layer(df, x=:time, y=:performance, group=:trial, Geom.line,style(line_width=0.5pt,default_color=lightColor)),
        Theme(background_color=colorant"white"),
        Guide.xlabel("generations"),
        Guide.ylabel(yLabel),
        Coord.Cartesian(ymin=0.0,ymax=1.0)
     )

    draw(PNG(filename, 2.5inch, 2inch),plt)
    
end
#=
plotPropertyLines(expressMatrix,"oilm_express_vb.png",colorant"blue","e")
plotPropertyLines(composeMatrix,"oilm_compose_vb.png",colorant"orange","c")
plotPropertyLines(stableMatrix,"oilm_stable_vb.png" ,colorant"purple","s")
=#

bottleV=collect(bottle0:bottleStep:bottle1)

plotProperty(expressMatrix1,expressMatrix0,bottleV,"oilm_express_vb.png",colorant"blue","e")
plotProperty(composeMatrix1,composeMatrix0,bottleV,"oilm_compose_vb.png",colorant"orange","c")
plotProperty(stableMatrix1,  stableMatrix0,bottleV,"oilm_stable_vb.png",colorant"purple","s")

