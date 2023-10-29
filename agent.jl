
using Flux

mutable struct Agent
    bitN::Int64
    s2m::Chain
    m2s::Vector{Int64}
end

function makeAgent(bitN::Int,hiddenN::Int)
    Agent(n,Chain(Dense(bitN=>hiddenN,sigmoid),Dense(hiddenN=>bitN,sigmoid)), Vector{Int64}(undef, 2^bitN))
end

function makeAgent(n::Int)
    makeAgent(n,n)
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
        thisColEntropy=0.0

        for signalCol in 1:n
            p=0.0
            for rowC in 1:2^n
                if messageMatrix[messageCol,rowC]*signalMatrix[signalCol,rowC]==1
                    p+=1.0
                end
            end
            p/=2^(n-1)
            thisColEntropy+=calculateEntropy(p)
        end

        entropy+=thisColEntropy/n

    end

    n-entropy

end
        


    
    

