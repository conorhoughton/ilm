
using Flux

mutable struct Agent
    bitN::Int64
    m2s::Chain
    m2sTable::Vector{Int64}
end

function m2s(agent::Agent,message::Int64)
    messageV=v2BV(agent.bigN,message-1)    
    mV2I(agent.m2s(messageV))+1
end
    
function makeAgent(bitN::Int,hiddenN::Int)
    Agent(n,Chain(Dense(bitN=>hiddenN,sigmoid),Dense(hiddenN=>bitN,sigmoid)),Vector{Int64}(undef, 2^bitN))
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
        

function makeTable(agent::Agent,exemplars::Vector{Int64})

    for exemplar in exemplars
        probs=agent.m2s(v2BV(agent.bitN,exemplar-1))
        signal=bV2I(round.(Int64,agent.m2s(v2BV(agent.bitN,exemplar-1))))+1
        agent.m2sTable[exemplar]=signal
    end
end

function randomTable(bitN::Int64)
    table=randperm(2^bitN)
end
        
    
    

