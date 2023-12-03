
using Flux

mutable struct Agent
    bitN::Int64
    s2m::Chain
    m2s::Vector{Int64}
end

function makeAgent(bitN::Int,hiddenN::Int)
    Agent(bitN,Chain(Dense(bitN=>hiddenN,sigmoid),Dense(hiddenN=>bitN,sigmoid)), Vector{Int64}(undef, 2^bitN))
end


function makeAgentReLU(bitN::Int,hiddenN::Int)
    Agent(bitN,Chain(Dense(bitN=>hiddenN,relu),Dense(hiddenN=>bitN,sigmoid)), Vector{Int64}(undef, 2^bitN))
end


function makeAgent(bitN::Int)
    makeAgent(bitN,bitN)
end


function makeAgentReLU(bitN::Int)
    makeAgentReLU(bitN,bitN)
end


function s2m(signal::Vector{Int},agent::Agent)
    agent.s2m(signal)
end


function s2m(agent::Agent,signal::Vector{Int})
    agent.s2m(signal)
end


function obvert(agent::Agent)

    n=2^agent.bitN

    probabilityTable=Matrix{Float64}(undef,n,n)
    
    for signal in 0:n-1
        
        nnSignal=s2m(agent,valueToBinaryVector(agent.bitN,signal))
        
        for message in 0:n-1
            messageVector=valueToBinaryVector(agent.bitN,message)
            probabilityTable[signal+1,message+1]=probabilityOfM(messageVector,nnSignal)
        end
    end

    for message in 1:n
        agent.m2s[message]=argmax(probabilityTable[:,message])
    end

    
    
end


function expressivity(agent::Agent)
    n=2^agent.bitN
    onto=zeros(Int64,n)
    for i in 1:n
        onto[agent.m2s[i]]=1
    end
    sum(onto)/n
end

function stability(table1::Vector,table2::Vector)
    total=0.0
    for i in 1:length(table1)
        if table1[i]==table2[i]
            total+=1
        end
    end
    total/length(table1)
end



function compositionality(agent::Agent)
    
    n=agent.bitN

    messageMatrix=Matrix{Int64}(undef,n,2^n)
    signalMatrix =Matrix{Int64}(undef,n,2^n)

    for messageC in 1:2^n
        message=v2BV(n,messageC-1)
        signal=v2BV(n,agent.m2s[messageC]-1)
        for i in 1:n
            messageMatrix[i,messageC]=message[i]
            signalMatrix[i,messageC]=signal[i]
        end
    end

    entropy=0.0

    for messageCol in 1:n
        
        thisColEntropy=zeros(Float64,n)
        
        for signalCol in 1:n
            p=0.0
            for rowC in 1:2^n
                if messageMatrix[messageCol,rowC]*signalMatrix[signalCol,rowC]==1
                    p+=1.0
                end
            end
            p/=2^(n-1)
            thisColEntropy[signalCol]=calculateEntropy(p)
        end

        entropy+=minimum(thisColEntropy)

    end

    1-entropy/n

end


function newCompose(agent::Agent)

    n=agent.bitN
    
    messageMatrix=Matrix{Int64}(undef,n,2^n)
    signalMatrix =Matrix{Int64}(undef,n,2^n)

    for messageC in 1:2^n
        message=v2BV(n,messageC-1)
        signal =v2BV(n,agent.m2s[messageC]-1)
        for i in 1:n
            messageMatrix[i,messageC]=message[i]
            signalMatrix[i,messageC]=signal[i]
        end
    end

    entropy=0.0

    signalColV=collect(1:n)

    entropyV=[Vector{Float64}() for _ in 1:n]
    
    for messageCol in 1:n

        thisColEntropy=ones(Float64,n)
        
        for signalCol in 1:n
            p=0.0
            for rowC in 1:2^n
                if messageMatrix[messageCol,rowC]*signalMatrix[signalCol,rowC]==1
                    p+=1.0
                end
            end
            p/=2^(n-1)
            thisColEntropy[signalCol]=calculateEntropy(p)
        end

        minVal, minIndex = findmin(thisColEntropy)

        append!(entropyV[minIndex],minVal)
        
    end

    print(entropyV)

    entropy=0.0
    
    for i in 1:n
        if length(entropyV[i])>0
            entropy+=minimum(entropyV[i])
        else
            entropy+=1
        end
    end

    1-entropy/n
    
end



    
    

