
using Flux

mutable struct Agent
    bitN::Int64
    m2s::Chain
end

function makeAgent(bitN::Int,hiddenN::Int)
    Agent(n,Chain(Dense(bitN=>hiddenN,sigmoid),Dense(hiddenN=>bitN,sigmoid)))
end

function makeAgent(n::Int)
    makeAgent(n,n)
end


function expressivity(agent::Agent)
    n=2^agent.bitN
    onto=zeros(Int64,n)
    for i in 1:n
        signal=agent.m2s(v2BV(agent.bitN,i))
        signal=round.(Int64,signal)
        signal=bV2I(signal)
        onto[signal+1]=1
    end
    sum(onto)/n
end
        


    
    

