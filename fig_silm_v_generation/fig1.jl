#=
makes a simple agent
trains it
records the performance against generation
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

generationN=40

trialsN=50

expressMatrix=Matrix{Float64}(undef,generationN,trialsN)
composeMatrix=Matrix{Float64}(undef,generationN,trialsN)



for trialC in 1:trialsN

    child = makeAgent(n)
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

        obvert(child)
        
        parent=copy(child.m2s)
    
        #println(generation," ",expressivity(child)," ",compositionality(child))
    
    end

end
        
expressMu=vec(mean(expressMatrix, dims=2))
expressStdDev = vec(std(expressMatrix, dims=2))

expressMin = expressMu .- expressStdDev
expressMax = expressMu .+ expressStdDev

generations=collect(0:generationN-1)

expressDf = DataFrame(X=generations, Y=expressMu, StdDev=expressStdDev,YMin=expressMin,YMax=expressMax)

plt=plot(layer(expressDf,x=:X,y=:Y, Geom.line),layer(expressDf,x=:X,ymin=:YMin,ymax=:YMax,Geom.ribbon),Theme(background_color=colorant"white"))

draw(PNG("fig1_express.png", 6inch, 4inch),plt)

       
composeMu=vec(mean(composeMatrix, dims=2))
composeStdDev = vec(std(composeMatrix, dims=2))

composeMin = composeMu .- composeStdDev
composeMax = composeMu .+ composeStdDev

composeDf = DataFrame(X=generations, Y=composeMu, StdDev=composeStdDev,YMin=composeMin,YMax=composeMax)


plt=plot(layer(composeDf,x=:X,y=:Y, Geom.line),layer(composeDf,x=:X,ymin=:YMin,ymax=:YMax,Geom.ribbon),Theme(background_color=colorant"white",default_color="orange"))


draw(PNG("fig1_compose.png", 6inch, 4inch), plt)

